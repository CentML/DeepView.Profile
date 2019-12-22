import logging
from concurrent.futures import ThreadPoolExecutor

from innpv.analysis.runner import analyze_project
from innpv.config import Config
from innpv.exceptions import AnalysisError
from innpv.nvml import NVML
import innpv.protocol_gen.innpv_pb2 as pm

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

    def submit_request(self, analysis_request, address):
        state = self._connection_manager.get_connection_state(address)
        if not state.is_request_current(analysis_request):
            logger.debug(
                'Ignoring stale analysis request %d from (%s:%d).',
                analysis_request.sequence_number,
                *address,
            )
            return

        state.update_sequence(analysis_request)

        if analysis_request.mock_response:
            self._handle_mock_analysis_request(analysis_request, address)
            return

        self._executor.submit(
            self._handle_analysis_request,
            analysis_request,
            state,
            address,
        )

    def _handle_analysis_request(self, analysis_request, state, address):
        try:
            logger.debug(
                'Processing request %d from (%s:%d).',
                analysis_request.sequence_number,
                *address,
            )
            analyzer = analyze_project(
                Config.project_root, Config.entry_point, self._nvml)

            memory_usage = next(analyzer)
            self._send_memory_usage_response(
                memory_usage, analysis_request.sequence_number, address)

        except AnalysisError as ex:
            self._enqueue_response(
                self._send_analysis_error,
                ex,
                analysis_request.sequence_number,
                address,
            )

        except:
            logger.exception(
                'Exception occurred when handling analysis request.')

    def _handle_mock_analysis_request(self, analysis_request, address):
        memory_usage = pm.MemoryUsageResponse()
        memory_usage.peak_usage_bytes = 1337
        memory_usage.memory_capacity_bytes = 13337
        self._message_sender.send_memory_usage_response(
            memory_usage,
            analysis_request.sequence_number,
            address,
        )

    def _send_memory_usage_response(
        self,
        memory_usage,
        sequence_number,
        address,
    ):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_memory_usage_response(
                memory_usage,
                sequence_number,
                address,
            )
        except:
            logger.exception(
                'Exception occurred when sending a memory usage response.')

    def _send_analysis_error(self, exception, sequence_number, address):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_analysis_error(
                str(exception), sequence_number, address)
        except:
            logger.exception(
                'Exception occurred when sending an analysis error.')
