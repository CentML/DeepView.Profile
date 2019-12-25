import logging

from innpv.legacy_analysis.parser import parse_source_code, analyze_code
from innpv.exceptions import AnalysisError

import innpv.protocol_gen.innpv_pb2 as pm

logger = logging.getLogger(__name__)


class MessageHandler:
    def __init__(
        self,
        connection_manager,
        message_sender,
        analysis_request_manager,
    ):
        self._connection_manager = connection_manager
        self._message_sender = message_sender
        self._analysis_request_manager = analysis_request_manager

    def _handle_initialize_request(self, message, address):
        state = self._connection_manager.get_connection_state(address)
        if state.initialized:
            self._message_sender.send_protocol_error(
                pm.ProtocolError.ErrorCode.ALREADY_INITIALIZED_CONNECTION,
                address,
            )
            return
        if message.protocol_version != 1:
            # We only support version 1 of the protocol.
            self._message_sender.send_protocol_error(
                pm.ProtocolError.ErrorCode.UNSUPPORTED_PROTOCOL_VERSION,
                address,
            )
            self._connection_manager.remove_connection(address)
            return

        state.initialized = True
        self._message_sender.send_initialize_response(address)

    def _handle_analysis_request(self, message, address):
        state = self._connection_manager.get_connection_state(address)
        if not state.initialized:
            self._message_sender.send_protocol_error(
                pm.ProtocolError.ErrorCode.UNINITIALIZED_CONNECTION,
                address,
            )
            return

        self._analysis_request_manager.submit_request(message, address)

    def handle_message(self, raw_data, address):
        try:
            message = pm.FromClient()
            message.ParseFromString(raw_data)
            logger.debug('Received message from (%s:%d).', *address)

            message_type = message.WhichOneof('payload')
            if message_type is None:
                logger.warn('Received empty message from (%s:%d).', *address)
                return

            if message_type == 'initialize':
                self._handle_initialize_request(
                    getattr(message, message_type), address)
            elif message_type == 'analysis':
                self._handle_analysis_request(
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
