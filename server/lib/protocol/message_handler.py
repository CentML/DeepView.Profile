import logging

from lib.analysis.parser import parse_source_code, analyze_code
from lib.exceptions import AnalysisError
import lib.models_gen.messages_pb2 as m

logger = logging.getLogger(__name__)


class MessageHandler:
    def __init__(self, message_sender, analysis_request_manager):
        self._message_sender = message_sender
        self._analysis_request_manager = analysis_request_manager

    def _handle_analyze_request(self, message, address):
        if message.mock_response:
            self._handle_analyze_request_mock_response(message, address)
            return
        self._analysis_request_manager.submit_request(message, address)

    def _handle_analyze_request_mock_response(self, message, address):
        """
        Return a mock response when requested to analyze a model definition.
        """
        try:
            _, annotation_info, model_operations = analyze_code(
                *parse_source_code(message.source_code))
            self._message_sender.send_mock_analyze_response(
                annotation_info,
                model_operations,
                message.sequence_number,
                address,
            )
        except AnalysisError as ex:
            self._message_sender.send_analyze_error(str(ex), address)

    def handle_message(self, raw_data, address):
        try:
            message = m.PluginMessage()
            message.ParseFromString(raw_data)
            logger.debug('Received message from (%s:%d).', *address)

            message_type = message.WhichOneof('payload')
            if message_type is None:
                logger.warn('Received empty message from (%s:%d).', *address)
                return

            if message_type == 'analyze_request':
                self._handle_analyze_request(
                    getattr(message, message_type), address)
            else:
                # If the protobuf was compiled properly, this block should
                # never be reached.
                raise AssertionError(
                    'Invalid message type "{}".'.format(message_type))
        except:
            logger.exception(
                'Processing message from (%s:%d) resulted in an exception.',
                *address,
            )
