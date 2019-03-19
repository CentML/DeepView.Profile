import logging
from concurrent.futures import ThreadPoolExecutor

from lib.analysis.parser import parse_source_code, analyze_code
from lib.exceptions import AnalysisError

logger = logging.getLogger(__name__)


class AnalysisRequestManager:
    def __init__(self, enqueue_response, message_sender):
        # NOTE: This counter is modified by a thread in the main executor, but
        #       will be read by other threads. No R/W lock is needed because of
        #       the Python GIL.
        #
        # NOTE: The sequence number from the client must be non-negative
        self._sequence_number = -1

        # NOTE: We don't actually get parallelism because of the GIL.
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Callable that enqueues a function onto the main executor
        self._enqueue_response = enqueue_response
        self._message_sender = message_sender

    def stop(self):
        self._executor.shutdown()

    def submit_request(self, analysis_request, address):
        if analysis_request.sequence_number <= self._sequence_number:
            return
        self._sequence_number = analysis_request.sequence_number
        self._executor.submit(
            self._handle_analysis_request,
            analysis_request,
            address,
        )

    def _is_request_current(self, analysis_request):
        return self._sequence_number <= analysis_request.sequence_number

    def _handle_analysis_request(self, analysis_request, address):
        """
        Process an analysis request with the ability to abort early if the
        request is no longer current.
        """
        try:
            tree, source_map = parse_source_code(analysis_request.source_code)
            if not self._is_request_current(analysis_request):
                return

            annotation_info, model_operations = analyze_code(tree, source_map)
            if not self._is_request_current(analysis_request):
                return

            self._enqueue_response(
                self._send_analysis_response,
                annotation_info,
                model_operations,
                address,
            )
        except AnalysisError as ex:
            self._enqueue_response(self._send_analysis_error, ex, address)
        except:
            logger.exception(
                'Exception occurred when handling analysis request.')

    def _send_analysis_response(
            self, annotation_info, model_operations, address):
        # Called from the main executor. Do not call directly!
        try:
            self._message_sender.send_mock_analyze_response(
                annotation_info, model_operations, address)
        except:
            logger.exception(
                'Exception occurred when sending an analysis response.')

    def _send_analysis_error(self, exception, address):
        try:
            # Called from the main executor. Do not call directly!
            self._message_sender.send_analyze_error(str(exception), address)
        except:
            logger.exception(
                'Exception occurred when sending an analysis error.')
