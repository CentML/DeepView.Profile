import logging
from concurrent.futures import ThreadPoolExecutor

from skyline.legacy_analysis.parser import parse_source_code, analyze_code
from skyline.legacy_analysis.request_cache import SourceCache, RuntimeCache
from skyline.profiler import to_trainable_model, get_performance_limits
from skyline.profiler.memory import get_memory_info
from skyline.profiler.module import get_operation_runtimes
from skyline.profiler.throughput import get_throughput_info
from skyline.exceptions import AnalysisError
from skyline.nvml import NVML

logger = logging.getLogger(__name__)


class AnalysisRequestManager:
    def __init__(self, enqueue_response, message_sender, connection_manager):
        # NOTE: We don't actually get parallelism because of the GIL.
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Callable that enqueues a function onto the main executor
        self._enqueue_response = enqueue_response
        self._message_sender = message_sender
        self._connection_manager = connection_manager
        self._nvml = NVML()

        self._source_cache = SourceCache()
        self._runtime_cache = RuntimeCache()

    def start(self):
        self._nvml.start()

    def stop(self):
        self._nvml.stop()
        self._executor.shutdown()

    def submit_request(self, analysis_request, address):
        state = self._connection_manager.get_connection_state(address)
        if not state.is_request_current(analysis_request):
            return
        state.update_sequence(analysis_request)
        self._executor.submit(
            self._handle_analysis_request,
            analysis_request,
            state,
            address,
        )

    def _is_request_current(self, analysis_request):
        return self._sequence_number <= analysis_request.sequence_number

    def _handle_analysis_request(self, analysis_request, state, address):
        """
        Process an analysis request with the ability to abort early if the
        request is no longer current.
        """
        try:
            logger.debug(
                'Processing request %d from (%s:%d).',
                analysis_request.sequence_number,
                *address,
            )

            tree, source_map = parse_source_code(analysis_request.source_code)
            if not state.is_request_current(analysis_request):
                return

            # 1. Parse the code to extract source locations
            class_name, annotation_info, model_operations = analyze_code(
                tree, source_map)
            if not state.is_request_current(analysis_request):
                return

            # 2. If the parse tree has not changed, use our cached response
            cached_results = self._source_cache.query(tree)
            if cached_results is not None:
                logger.debug(
                    'Using cached response for request %d from (%s:%d).',
                    analysis_request.sequence_number,
                    *address,
                )
                # cached_results[0] is the cached model_operations map
                model_operations.set_runtimes_from_cache(cached_results[0])
                self._enqueue_response(
                    self._send_analysis_response,
                    annotation_info,
                    model_operations,
                    *cached_results[1:],
                    analysis_request.sequence_number,
                    address,
                )
                return
            model = to_trainable_model(tree, class_name)
            if not state.is_request_current(analysis_request):
                return

            # 3. Profile the model layer by layer
            # NOTE: This function makes in-place changes to model_operations
            # NOTE: This function will attach hooks to the model
            get_operation_runtimes(
                model, annotation_info, model_operations, self._runtime_cache)
            if not state.is_request_current(analysis_request):
                return
            self._enqueue_response(
                self._send_profiled_layers_response,
                model_operations,
                analysis_request.sequence_number,
                address,
            )

            # NOTE: We need to re-instantiate the model because
            #       get_operation_runtimes() modifies it.
            del model
            model = to_trainable_model(tree, class_name)

            # 4. Profile the model's overall memory usage
            memory_info = get_memory_info(
                analysis_request.source_code,
                class_name,
                annotation_info,
                self._nvml,
            )
            if not state.is_request_current(analysis_request):
                return
            self._enqueue_response(
                self._send_memory_info_response,
                memory_info,
                annotation_info,
                analysis_request.sequence_number,
                address,
            )

            # 5. Profile the model's throughput
            throughput_info = get_throughput_info(
                model, annotation_info, memory_info)
            perf_limits = get_performance_limits(memory_info, throughput_info)
            self._enqueue_response(
                self._send_throughput_info_response,
                throughput_info,
                perf_limits,
                analysis_request.sequence_number,
                address,
            )

            # 6. Cache the overall results
            results = (
                model_operations,
                memory_info,
                throughput_info,
                perf_limits,
            )
            self._source_cache.store(tree, results)

        except AnalysisError as ex:
            # NOTE: Error responses are not cached
            self._enqueue_response(
                self._send_analysis_error,
                ex,
                analysis_request.sequence_number,
                address,
            )

        except:
            logger.exception(
                'Exception occurred when handling analysis request.')


    def _send_analysis_response(
        self,
        annotation_info,
        model_operations,
        memory_info,
        throughput_info,
        perf_limits,
        sequence_number,
        address,
    ):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_analyze_response(
                annotation_info,
                model_operations,
                memory_info,
                throughput_info,
                perf_limits,
                sequence_number,
                address,
            )
        except:
            logger.exception(
                'Exception occurred when sending an analysis response.')

    def _send_analysis_error(self, exception, sequence_number, address):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_analyze_error(
                str(exception), sequence_number, address)
        except:
            logger.exception(
                'Exception occurred when sending an analysis error.')

    def _send_profiled_layers_response(
            self, model_operations, sequence_number, address):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_profiled_layers_response(
                model_operations, sequence_number, address)
        except:
            logger.exception(
                'Exception occurred when sending a profiled layers response.')

    def _send_memory_info_response(
            self, memory_info, annotation_info, sequence_number, address):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_memory_info_response(
                memory_info, annotation_info, sequence_number, address)
        except:
            logger.exception(
                'Exception occurred when sending a memory info response.')

    def _send_throughput_info_response(
            self, throughput_info, perf_limits, sequence_number, address):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_throughput_info_response(
                throughput_info, perf_limits, sequence_number, address)
        except:
            logger.exception(
                'Exception occurred when sending a throughput info response.')
