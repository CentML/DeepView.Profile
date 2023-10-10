import subprocess
import functools
import orjson
import io
import traceback
import numpy as np
import logging
import time
import os
from collections import deque
from perfetto.trace_processor import TraceProcessor
from deepview_profile.user_code_utils import user_code_environment
from deepview_profile.exceptions import AnalysisError
from torch_tb_profiler.profiler.tensor_core import TC_Allowlist
import dill
from torch.profiler import profile, schedule, ProfilerActivity
import sys
logger = logging.getLogger("utilization")
logger.setLevel(logging.DEBUG)
FILENAME = "raw_trace_file.json"


class Node:
    def __init__(self, name, start, end, duration, track, depth, slice_id, cpu_forward, gpu_forward, cpu_backward, gpu_backward):
        self.name = name
        self.start = start
        self.end = end
        self.duration = duration
        self.track = track
        self.depth = depth
        self.slice_id = slice_id
        self.cpu_forward = cpu_forward
        self.gpu_forward = gpu_forward
        self.gpu_forward_span = 0
        self.cpu_backward = cpu_backward
        self.gpu_backward = gpu_backward
        self.cpu_backward_span = 0
        self.cpu_backward_slices = list()
        self.gpu_backward_span = 0
        self.gpu_backward_slices = list()
        self.children = list()

    def __str__(self):
        return f"{self.name} \t {self.slice_id} "

    def postorder(self):
        ret = []
        for c in self.children:
            ret.extend(c.postorder())
        ret.append(self)

        return ret


class UtilizationProfiler:
    def __init__(self, logging_level):
        self._root_node = None
        self._tensor_core_perc = None
        self._logging_level = logging_level

    def _calculate_gpu_times(self, kernel_list):
        # return (span, net time)
        if not kernel_list:
            return 0, 0
        start, end, netTime = kernel_list[0][0], kernel_list[0][1], kernel_list[0][1] - \
            kernel_list[0][0]

        for s, e in kernel_list[1:]:
            if start <= s < end and end < e:
                netTime += e - end
            elif end <= s:
                netTime += e - s
            end = max(end, e)

        return (end - start), netTime

    def _calculate_gpu_forward_time(self, tp, node):
        span = 0
        time = 0
        cudaLaunchList = tp.query_dict(f"""select * from slices where name like '%CudaLaunchKernel%' and track_id={node.track}
                                    and depth>{node.depth} and ts between {node.start} and {node.end} ORDER BY ts ASC""")

        # add individual kernels
        kernel_list = []
        for cudaLaunch in cudaLaunchList:
            slice_id_origin = cudaLaunch['slice_id']
            slice_id_destination = tp.query_dict(
                f"""select * from flow where slice_out={slice_id_origin}""")
            if slice_id_destination:
                cuda_slice = tp.query_dict(
                    f"select * from slice where slice_id={slice_id_destination[0]['slice_in']}")
                kernel_list.append(
                    (cuda_slice[0]['ts'], cuda_slice[0]['ts'] + cuda_slice[0]['dur']))

        if kernel_list:
            kernel_list.sort(key=lambda x: x[0])
            span, time = self._calculate_gpu_times(kernel_list)

        node.gpu_forward_span = span
        node.gpu_forward = time

    def _backward_slices(self, tp):
        res = []
        backwardTopOps = tp.query_dict(
            "select * from slices where name like '%Autograd::engine%' and depth=0 ORDER BY ts ASC")
        for bto in backwardTopOps:
            kernel_slices = []
            endTime = bto['ts'] + bto['dur']
            cudaLaunchList = tp.query_dict(f"""select * from slices where name like '%CudaLaunchKernel%' and track_id={bto['track_id']}
                                    and depth>{bto['depth']} and ts between {bto['ts']} and {endTime}""")

            for cudaLaunch in cudaLaunchList:
                slice_id_origin = cudaLaunch['slice_id']
                slice_id_destination = tp.query_dict(
                    f"""select * from flow where slice_out={slice_id_origin}""")
                if slice_id_destination:
                    cuda_slice = tp.query_dict(
                        f"select * from slice where slice_id={slice_id_destination[0]['slice_in']}")
                    kernel_slices.append(cuda_slice[0])

            bto['name'] = bto["name"].split()[-1]
            bto['kernel_list'] = kernel_slices
            res.append(bto)
        return res

    def _accumulate_backward_slices_to_node(self, node):
        for ch in node.children:
            node.cpu_backward_slices.extend(
                self._accumulate_backward_slices_to_node(ch))

        return node.cpu_backward_slices

    def _calculate_backward_cpu_span(self, node):
        backward_slices = node.cpu_backward_slices
        kernel_slices = []
        if not backward_slices:
            return
        minTs = float('inf')
        maxTs = float('-inf')
        netTime = 0
        backward_slices.sort(key=lambda x: x['ts'])
        for bw_slice in backward_slices:
            minTs = min(minTs, bw_slice['ts'])
            maxTs = max(maxTs, bw_slice['ts']+bw_slice['dur'])
            netTime += bw_slice['dur']
            kernel_slices.extend([(kernel['ts'], kernel['ts']+kernel['dur'])
                                 for kernel in bw_slice['kernel_list']])

        node.cpu_backward_span = maxTs - minTs
        node.cpu_backward = netTime
        if kernel_slices:
            kernel_slices.sort(key=lambda x: x[0])
            node.gpu_backward_span, node.gpu_backward = self._calculate_gpu_times(
                kernel_slices)

    def _populate_backward_data(self, node):
        self._calculate_backward_cpu_span(node)
        for ch in node.children:
            self._populate_backward_data(ch)

    @functools.lru_cache(maxsize=None)
    def _can_match(self, f, b):
        if "aten" in f and "Backward0" in b:
            raw_f = f[len("aten::"):].lower().replace("_", "")
            raw_b = b[:-len("Backward0")].lower()
            return raw_f == raw_b or raw_f == "transpose" and raw_b == 't' or \
                raw_f == 't' and raw_b == 'transpose'

        return False

    # solves the longest common subsequence (LCS) problem, matching
    # the post-ordered model forward pass with the sequence of backward
    # operators executed.
    #
    # this function returns a list of tuples whose elements are in the form:
    #  (forward, backward)

    def _lcs(self, forward, backward):
        N, M = len(forward), len(backward)
        dp = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                dp[i, j] = int(self._can_match(
                    forward[i].name, backward[j]['name']))
                if i > 0:
                    dp[i, j] = max(dp[i, j], dp[i-1, j])
                if j > 0:
                    dp[i, j] = max(dp[i, j], dp[i, j-1])
                if i > 0 and j > 0 and self._can_match(forward[i].name, backward[j]['name']):
                    dp[i, j] = max(dp[i, j], 1 + dp[i-1, j-1])

        matchings = []
        i, j = N-1, M-1
        while i > 0 and j > 0:
            if self._can_match(forward[i].name, backward[j]['name']):
                matchings.append((forward[i], backward[j]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i-1, j] == dp[i, j]:
                i -= 1
            else:
                j -= 1

        if self._logging_level == logging.DEBUG:
            logger.debug(
                f"N = {N}, M = {M}, best matching: {len(matchings)}\n")
        return matchings

    def _convert_ids_int_string(self, slices):
        for slice in slices:
            if 'id' in slice:
                slice['id'] = str(slice['id'])

    def _convert_negative_tids_to_positive(self, slices):
        for slice in slices:
            if 'tid' in slice and isinstance(slice['tid'], int):
                slice['tid'] = abs(slice['tid'])

    def _remove_args(self, slices):
        [slice.pop('args', None) for slice in slices]

    def _filter_traces(self, raw_slices):
        names_to_filter = ['profiler.py', 'built-in', "torch/", "cudaDeviceSynchronize",
                           "typing.py", "<module>", "os.py", "_collections", "enum.py", "numpy/", 'DataParallel', 'lib/', '.py']
        idx_to_filter = []
        for idx, item in enumerate(raw_slices['traceEvents']):
            # SKIP HIGH LEVEL TRACE IDENTIFIER FOR BACKWARD PASS
            if item['name'].startswith('torch/autograd') and item['name'].endswith('backward'):
                continue
            for keyword in names_to_filter:
                if keyword in item['name']:
                    idx_to_filter.append(idx)
            # SPECIFIC CASE
            if item['name'].startswith('entry_point') and item['name'].endswith('iteration'):
                idx_to_filter.append(idx)
            if item['name'].startswith('entry_point') and item['name'].endswith('forward'):
                idx_to_filter.append(idx)

        filtered_slices = [event for idx, event in enumerate(
            raw_slices['traceEvents']) if idx not in idx_to_filter]
        raw_slices['traceEvents'] = filtered_slices

        if self._logging_level == logging.DEBUG:
            with open("filtered_slices.json", 'wb') as f:
                f.write(orjson.dumps(raw_slices))

    def _get_perfetto_object(self, filepath):
        with open(filepath, 'rb') as f:
            raw_slices = orjson.loads(f.read())

        self._filter_traces(raw_slices)

        if isinstance(raw_slices, dict) and 'traceEvents' in raw_slices:
            # Perfetto doesn't want this format produced by PyTorch
            raw_slices = raw_slices.pop('traceEvents', None)
        # Convert IDs from int to string. Without this perfetto fails to JSON load trace with IDs stored as integers.
        self._convert_ids_int_string(raw_slices)
        # Convert negative 'tid' values to positive. Without this perfetto combines together the slices with different tids into one track
        self._convert_negative_tids_to_positive(raw_slices)
        self._remove_args(raw_slices)  # For speedup

        slices_bytes = orjson.dumps(raw_slices)
        slices_bytes = io.BytesIO(slices_bytes)  # 4x speedup using orjson

        try:
            tp = TraceProcessor(slices_bytes)
        except ConnectionResetError as e:
            # This happens sometimes so retry once
            tp = TraceProcessor(slices_bytes)

        interesting_fields = 'SELECT ts, dur, track_id, category, name, depth, cat, slice_id, id, arg_set_id FROM slice'

        def query_dict(query):
            query = query.lower().replace('select * from slice',
                                          interesting_fields)  # gives a 15% speedup
            try:
                query_iterator = tp.query(query)
            except Exception as e:
                print('[ERROR] Unable to run query: %s' % query)
                print(traceback.format_exc())
                print('\n\n')
                raise e
            return [item.__dict__ for item in query_iterator]

        tp.query_dict = query_dict

        return tp

    def _convert_node_to_dict(self, node):
        newEntry = {'slice_id': node.slice_id,
                    'name': node.name,
                    'start': node.start,
                    'end': node.end,
                    'cpu_forward': node.duration,
                    'cpu_forward_span': node.cpu_forward,
                    'gpu_forward': node.gpu_forward,
                    'gpu_forward_span': node.gpu_forward_span,
                    'cpu_backward': node.cpu_backward,
                    'cpu_backward_span': node.cpu_backward_span,
                    'gpu_backward': node.gpu_backward,
                    'gpu_backward_span': node.gpu_backward_span,
                    'children': list()}
        for ch in node.children:
            newEntry['children'].append(self._convert_node_to_dict(ch))
        return newEntry

    def output_to_json(self, path_save_file):
        node_json = self._convert_node_to_dict(self._root_node)
        output = {'node': node_json, 'tensor_core': self._tensor_core_perc}
        with open(os.path.join(path_save_file, 'profiling_results.json'), 'wb') as f:
            f.write(orjson.dumps(output))

    def _calculate_tensor_core_utilization(self, filepath):
        kernelDict = {"tensorTime": 0, "noTensorTime": 0, "totalTime": 0}
        with open(filepath, 'r') as f:
            data = orjson.loads(f.read())
        for event in data["traceEvents"]:
            if event.get("cat") and event["cat"] == "kernel":
                if event["name"] in TC_Allowlist:
                    kernelDict["tensorTime"] += event["dur"]
                else:
                    kernelDict["noTensorTime"] += event["dur"]

        totalTime = kernelDict["tensorTime"] + kernelDict["noTensorTime"]
        if self._logging_level == logging.DEBUG:
            logger.debug(
                f'Tensor time: {kernelDict["tensorTime"]} perc {round(kernelDict["tensorTime"]/totalTime*100,2)}\n')
            logger.debug(
                f'No Tensor time: {kernelDict["noTensorTime"]} perc {round(kernelDict["noTensorTime"]/totalTime*100,2)}\n')
            logger.debug(f'totalTime: {totalTime}\n')
        return round(kernelDict["tensorTime"]/totalTime*100, 2)

    def _deepview_analysis(self, filepath):
        startTime = time.time()
        tp = self._get_perfetto_object(filepath)

        profilerStepStart = tp.query_dict(
            "select * from slices where name like '%ProfilerStep%'")
        main_track = profilerStepStart[0]["track_id"]
        start = profilerStepStart[0]["ts"]
        end = start + profilerStepStart[0]["dur"]
        profilerStartDepth = profilerStepStart[0]["depth"]

        rootQuery = tp.query_dict(f"""
                                select * from slices where name like '%nn.Module:%' 
                                and depth = 
                                (SELECT MIN(depth) from slices where name like '%nn.Module%' and depth>{profilerStartDepth} and ts between {start} and {end} and track_id = {main_track})                
                                """)[0]

        rootNode = Node(rootQuery['name'], rootQuery['ts'], rootQuery['ts']+rootQuery['dur'],
                        rootQuery['dur'], rootQuery['track_id'], rootQuery['depth'], rootQuery['slice_id'], rootQuery['dur'], 0, 0, 0)
        self._calculate_gpu_forward_time(tp, rootNode)
        logger.debug(f"{rootNode}\n")

        stack = deque([rootNode])

        while stack:
            node = stack.popleft()
            queryNewDepth = tp.query_dict(f"""select MIN(depth) from slices where (name like '%nn.Module:%' or name like '%aten::%') and 
                                        depth>{node.depth} and track_id={node.track}
                                        and ts between {node.start} and {node.end}""")
            minDepth = queryNewDepth[0]['min(depth)'] if queryNewDepth else -1
            if minDepth:
                queryResults = tp.query_dict(f"""select * from slices where (name like '%nn.Module:%' or name like '%aten::%') 
                                        and depth={minDepth} and track_id={node.track} 
                                            and ts between {node.start} and {node.end}""")
                for qr in queryResults:
                    newNode = Node(qr['name'], qr['ts'], qr['ts']+qr['dur'],
                                   qr['dur'], qr['track_id'], qr['depth'], qr['slice_id'], qr['dur'], 0, 0, 0)
                    self._calculate_gpu_forward_time(tp, newNode)
                    node.children.append(newNode)
                    stack.append(newNode)

        ForwardPostOrderTraversal = list(
            filter(lambda x: 'aten::' in x.name, rootNode.postorder()))
        ForwardPostOrderTraversal.sort(key=lambda x: x.start, reverse=True)
        backwardSlices = self._backward_slices(tp)
        matchings = self._lcs(ForwardPostOrderTraversal, backwardSlices)

        countAccumulateGrad = 0
        for slice in backwardSlices:
            if 'AccumulateGrad' in slice['name']:
                countAccumulateGrad += 1

        numValidBackwardSlices = len(backwardSlices) - countAccumulateGrad
        if self._logging_level == logging.DEBUG:
            logger.debug(f'number of valid slices: {numValidBackwardSlices}\n')
            logger.debug(
                f'Number of matched slices: {len(matchings)} percentage: {round(len(matchings)/numValidBackwardSlices*100,2)}\n')

        for match in matchings:
            node = match[0]
            backward_slice = match[1]
            node.cpu_backward_slices.append(backward_slice)

        ## DEBUGGING PROCESS BACKWARD SLICES MATCHING #############
        if self._logging_level == logging.DEBUG:
            matchings.sort(key=lambda x: x[1]['ts'])
            with open('matching_results.txt', 'w') as file:
                file.write(
                    f'Total number of backward slices: {len(backwardSlices)}\n')
                file.write(
                    f'number of valid slices: {numValidBackwardSlices}\n')
                file.write(
                    f'Number of matched slices: {len(matchings)} percentage: {round(len(matchings)/numValidBackwardSlices*100,2)}\n')

                for m in matchings:
                    file.write(
                        f"forward: {m[0].name} {m[0].slice_id} backward: {m[1]['name']} {m[1]['slice_id']}\n")

        self._accumulate_backward_slices_to_node(rootNode)
        self._populate_backward_data(rootNode)

        if self._logging_level == logging.DEBUG:
            profilingResult = self._convert_node_to_dict(rootNode)
            with open('profiling_results.json', 'wb') as f:
                f.write(orjson.dumps(profilingResult))

        endTime = time.time()
        logger.debug(f'Total elapsed time: {endTime - startTime}')
        self._root_node = rootNode
        self._tensor_core_perc = self._calculate_tensor_core_utilization(
            filepath)

        # DELETE PYTORCH PROFILER TRACE (except for debugging purposes)
        if not (self._logging_level == logging.DEBUG):
            subprocess.run(["rm", '-f', os.path.join(os.getcwd(), filepath)])


def _trace_handler(p):
    p.export_chrome_trace(FILENAME)


def _serialize_node(respNode, internalNode):
    respNode.slice_id = internalNode["slice_id"]
    respNode.name = internalNode["name"]
    respNode.start = int(internalNode["start"])
    respNode.end = int(internalNode["end"])
    respNode.cpu_forward = int(internalNode["cpu_forward"])
    respNode.cpu_forward_span = int(internalNode["cpu_forward_span"])
    respNode.gpu_forward = int(internalNode["gpu_forward"])
    respNode.gpu_forward_span = int(internalNode["gpu_forward_span"])
    respNode.cpu_backward = int(internalNode["cpu_backward"])
    respNode.cpu_backward_span = int(internalNode["cpu_backward_span"])
    respNode.gpu_backward = int(internalNode["gpu_backward"])
    respNode.gpu_backward_span = int(internalNode["gpu_backward_span"])

    for ch in internalNode["children"]:
        addRespNode = respNode.children.add()
        _serialize_node(addRespNode, ch)


def serialize_response(respNode, rootNode):
    _serialize_node(respNode, rootNode)


def utilization_analysis(queue, payload, path_to_entry_point_dir):
    sys.path.append(path_to_entry_point_dir)
    try:
        model_provider, input_provider, iteration_provider, logging_level = dill.loads(
            payload)
        model = model_provider()
        inputs = input_provider()
        iteration = iteration_provider(model)
        skip_first = 2
        wait = 1
        warmup = 1
        active = 1
        totalIterations = skip_first + wait + warmup + active
        deepviewSchedule = schedule(
            skip_first=skip_first, wait=wait, warmup=warmup, active=active, repeat=1)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     schedule=deepviewSchedule,
                     on_trace_ready=_trace_handler,
                     with_stack=True) as p:
            for _ in range(totalIterations):
                iteration(*inputs)
                p.step()

        utilization = UtilizationProfiler(logging_level)
        path_to_file = os.path.join(os.getcwd(), FILENAME)
        utilization._deepview_analysis(path_to_file)
        jsonFormat = {"root_node": utilization._convert_node_to_dict(
            utilization._root_node), "tensor_core_perc": utilization._tensor_core_perc}
        queue.put(jsonFormat)
    except AnalysisError as ex:
        message = str(ex)
        logger.error(message)
        queue.put({"error": message})
    except Exception as ex:
        message = str(ex)
        logger.error(message)
        queue.put({"error": message})
