import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor

from skyline.analysis.runner import analyze_project
from skyline.config import Config
from skyline.exceptions import AnalysisError
from skyline.nvml import NVML
import skyline.protocol_gen.innpv_pb2 as pm

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
        start_time = time.perf_counter()
        try:
            logger.debug(
                'Processing request %d from (%s:%d).',
                context.sequence_number,
                *(context.address),
            )
            analyzer = analyze_project(
                Config.project_root, Config.entry_point, self._nvml)

            # Abort early if the connection has been closed
            if not context.state.connected:
                logger.debug(
                    'Aborting request %d from (%s:%d) early '
                    'because the client has disconnected.',
                    context.sequence_number,
                    *(context.address),
                )
                return

            breakdown = next(analyzer)
            self._enqueue_response(
                self._send_breakdown_response,
                breakdown,
                context,
            )

            if not context.state.connected:
                logger.debug(
                    'Aborting request %d from (%s:%d) early '
                    'because the client has disconnected.',
                    context.sequence_number,
                    *(context.address),
                )
                return

            throughput = next(analyzer)
            self._enqueue_response(
                self._send_throughput_response,
                throughput,
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

        except:
            logger.exception(
                'Exception occurred when handling analysis request.')
            self._enqueue_response(
                self._send_analysis_error,
                AnalysisError(
                    'An unexpected error occurred when analyzing your model. '
                    'Please file a bug report and then restart Skyline.'
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
        except:
            logger.exception(
                'Exception occurred when sending a breakdown response.')

    def _send_analysis_error(self, exception, context):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_analysis_error(exception, context)
        except:
            logger.exception(
                'Exception occurred when sending an analysis error.')

    def _send_throughput_response(self, throughput, context):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_throughput_response(throughput, context)
        except:
            logger.exception(
                'Exception occurred when sending a throughput response.')
