import os
import logging
from random import random

from skyline.config import Config
from skyline.exceptions import NoConnectionError

import skyline.protocol_gen.innpv_pb2 as pm

logger = logging.getLogger(__name__)


class MessageSender:
    def __init__(self, connection_manager):
        self._connection_manager = connection_manager

    def send_initialize_response(self, context):
        message = pm.InitializeResponse()
        message.server_project_root = Config.project_root
        message.entry_point.components.extend(Config.entry_point.split(os.sep))
        self._send_message(message, 'initialize', context)

    def send_protocol_error(self, error_code, context):
        message = pm.ProtocolError()
        message.error_code = error_code
        self._send_message(message, 'error', context)

    def send_breakdown_response(self, breakdown, context):
        # Ideally, MessageSender users should not need to know about the INNPV
        # protocol messages. However, to avoid extraneous copies, sometimes
        # callers will pass in constructed messages for sending.
        self._send_message(breakdown, 'breakdown', context)

    def send_analysis_error(self, analysis_error, context):
        message = pm.AnalysisError()
        message.error_message = str(analysis_error)
        if analysis_error.file_context is not None:
            message.file_context.file_path.components.extend(
                analysis_error.file_context.file_path.split(os.sep)
            )
            message.file_context.line_number = (
                analysis_error.file_context.line_number
                if analysis_error.file_context.line_number is not None
                else 0
            )
        self._send_message(message, 'analysis_error', context)

    def send_throughput_response(self, throughput, context):
        self._send_message(throughput, 'throughput', context)

    def _send_message(self, message, payload_name, context):
        try:
            connection = self._connection_manager.get_connection(
                context.address)
            enclosing_message = pm.FromServer()
            getattr(enclosing_message, payload_name).CopyFrom(message)
            enclosing_message.sequence_number = context.sequence_number
            connection.send_bytes(enclosing_message.SerializeToString())
        except NoConnectionError:
            logger.debug(
                'Not sending message to (%s:%d) because it is no longer '
                'connected.',
                *context.address,
            )
