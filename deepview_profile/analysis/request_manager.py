import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
from deepview_profile.analysis.runner import analyze_project
from deepview_profile.exceptions import AnalysisError
from deepview_profile.nvml import NVML
import deepview_profile.protocol_gen.innpv_pb2 as pm

logger = logging.getLogger(__name__)


class AnalysisRequestManager:
    def __init__(self, enqueue_response, message_sender, connection_manager):
        # We use a separate thread to perform the analysis request to not block
        # the server thread from handling messages from other connections.
        #
        # NOTE: We don't actually get parallelism because of the GIL.
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Callable that enqueues a function onto the main executor
        self._enqueue_response = enqueue_response
        self._message_sender = message_sender
        self._connection_manager = connection_manager
        self._nvml = NVML()
        mp.set_start_method("spawn")

    def start(self):
        self._nvml.start()

    def stop(self):
        self._nvml.stop()
        self._executor.shutdown()

    def submit_request(self, analysis_request, context):
        if analysis_request.mock_response:
            self._handle_mock_analysis_request(analysis_request, context)
            return

        self._executor.submit(
            self._handle_analysis_request,
            analysis_request,
            context,
        )

    def _handle_analysis_request(self, analysis_request, context):
        print("handle_analysis_request: begin")
        start_time = time.perf_counter()
        try:
            logger.debug(
                'Processing request %d from (%s:%d).',
                context.sequence_number,
                *(context.address),
            )
            connection = self._connection_manager.get_connection(
                context.address)
            analyzer = analyze_project(
                connection.project_root, connection.entry_point, self._nvml)

            # Abort early if the connection has been closed
            if self._early_disconnection_error(context):
                return

            breakdown = next(analyzer)
            self._enqueue_response(
                self._send_breakdown_response,
                breakdown,
                context,
            )

            if self._early_disconnection_error(context):
                return

            throughput = next(analyzer)
            self._enqueue_response(
                self._send_throughput_response,
                throughput,
                context,
            )

            # send habitat response
            if self._early_disconnection_error(context):
                return

            habitat_resp = next(analyzer)
            self._enqueue_response(
                self._send_habitat_response,
                habitat_resp,
                context,
            )

            # send utilization data
            if self._early_disconnection_error(context):
                return

            utilization_resp = next(analyzer)
            self._enqueue_response(
                self._send_utilization_response,
                utilization_resp,
                context
            )

            # send energy response
            if self._early_disconnection_error(context):
                return

            energy_resp = next(analyzer)
            self._enqueue_response(
                self._send_energy_response,
                energy_resp,
                context,
            )

            elapsed_time = time.perf_counter() - start_time
            logger.debug(
                'Processed analysis request %d from (%s:%d) in %.4f seconds.',
                context.sequence_number,
                *(context.address),
                elapsed_time,
            )

        except AnalysisError as ex:
            self._enqueue_response(self._send_analysis_error, ex, context)

        except Exception:
            logger.exception(
                'Exception occurred when handling analysis request.')
            self._enqueue_response(
                self._send_analysis_error,
                AnalysisError(
                    'An unexpected error occurred when analyzing your model. '
                    'Please file a bug report and then restart DeepView.'
                ),
                context,
            )

    def _handle_mock_analysis_request(self, analysis_request, context):
        # This runs on the main executor
        breakdown = pm.BreakdownResponse()
        breakdown.peak_usage_bytes = 1337
        breakdown.memory_capacity_bytes = 13337
        breakdown.iteration_run_time_ms = 133.7
        self._message_sender.send_breakdown_response(breakdown, context)

        throughput = pm.ThroughputResponse()
        throughput.samples_per_second = 1337
        throughput.predicted_max_samples_per_second = math.nan
        self._message_sender.send_throughput_response(throughput, context)

    def _send_breakdown_response(self, breakdown, context):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_breakdown_response(breakdown, context)
        except Exception:
            logger.exception(
                'Exception occurred when sending a breakdown response.')

    def _send_analysis_error(self, exception, context):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_analysis_error(exception, context)
        except Exception:
            logger.exception(
                'Exception occurred when sending an analysis error.')

    def _send_throughput_response(self, throughput, context):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_throughput_response(throughput, context)
        except Exception:
            logger.exception(
                'Exception occurred when sending a throughput response.')

    def _send_habitat_response(self, habitat_resp, context):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_habitat_response(habitat_resp, context)
        except Exception:
            logger.exception(
                'Exception occurred when sending a DeepView.Predict response.')

    def _send_energy_response(self, energy_resp, context):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_energy_response(energy_resp, context)
        except Exception:
            logger.exception(
                'Exception occurred when sending an energy response.')

    def _send_utilization_response(self, utilization_resp, context):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_utilization_response(
                utilization_resp, context)
        except Exception:
            logger.exception(
                'Exception occurred when sending utilization response.')

    def _early_disconnection_error(self, context):
        if not context.state.connected:
            logger.error(
                'Aborting request %d from (%s:%d) early '
                'because the client has disconnected.',
                context.sequence_number,
                *(context.address),
            )
            return True

        return False
