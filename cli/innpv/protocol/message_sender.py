import os
from random import random

from innpv.config import Config
from innpv.protocol.error_codes import ErrorCode

import innpv.protocol_gen.innpv_pb2 as pm


class MessageSender:
    def __init__(self, connection_manager):
        self._connection_manager = connection_manager

    def send_initialize_response(self, address):
        message = pm.InitializeResponse()
        message.server_project_root = Config.project_root
        message.entry_point.components.extend(Config.entry_point.split(os.sep))
        self._send_message(message, 'initialize', address)

    def send_protocol_error(self, error_code: ErrorCode, address):
        message = pm.ProtocolError()
        message.error_code = error_code.value
        self._send_message(message, 'error', address)

    def _send_message(self, message, payload_name, address):
        connection = self._connection_manager.get_connection(address)
        enclosing_message = pm.FromServer()
        getattr(enclosing_message, payload_name).CopyFrom(message)
        connection.send_bytes(enclosing_message.SerializeToString())
