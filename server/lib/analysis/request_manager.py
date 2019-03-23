import logging
from concurrent.futures import ThreadPoolExecutor

from lib.analysis.parser import parse_source_code, analyze_code
from lib.profiler import to_trainable_model, get_performance_limits
from lib.profiler.memory import get_memory_info
from lib.profiler.module import get_operation_runtimes
from lib.profiler.throughput import get_throughput_info
from lib.exceptions import AnalysisError
from lib.nvml import NVML

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

            # If the parse tree has not changed, use our cached response
            cached_results = state.source_cache.query(tree)
            if cached_results is not None:
                logger.debug(
                    'Using cached response for request %d from (%s:%d).',
                    analysis_request.sequence_number,
                    *address,
                )
                self._enqueue_response(
                    self._send_analysis_response,
                    *cached_results,
                    analysis_request.sequence_number,
                    address,
                )
                return

            class_name, annotation_info, model_operations = analyze_code(
                tree, source_map)
            if not state.is_request_current(analysis_request):
                return

            model = to_trainable_model(tree, class_name)
            if not state.is_request_current(analysis_request):
                return

            memory_info = get_memory_info(
                analysis_request.source_code,
                class_name,
                annotation_info,
                self._nvml,
            )
            if not state.is_request_current(analysis_request):
                return

            throughput_info = get_throughput_info(
                model, annotation_info, memory_info)
            if not state.is_request_current(analysis_request):
                return

            perf_limits = get_performance_limits(memory_info, throughput_info)
            if not state.is_request_current(analysis_request):
                return

            # This function makes in-place changes to model_operations
            get_operation_runtimes(
                model, annotation_info, model_operations, state.runtime_cache)

            results = (
                annotation_info,
                model_operations,
                memory_info,
                throughput_info,
                perf_limits,
            )
            state.source_cache.store(tree, results)

            self._enqueue_response(
                self._send_analysis_response,
                *results,
                analysis_request.sequence_number,
                address,
            )

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
        try:
            # Called from the main executor. Do not call directly!
            self._message_sender.send_analyze_error(
                str(exception), sequence_number, address)
        except:
            logger.exception(
                'Exception occurred when sending an analysis error.')
