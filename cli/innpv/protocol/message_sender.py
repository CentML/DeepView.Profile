import os
from random import random

from innpv.config import Config

import innpv.protocol_gen.innpv_pb2 as pm


class MessageSender:
    def __init__(self, connection_manager):
        self._connection_manager = connection_manager

    def send_initialize_response(self, address):
        message = pm.InitializeResponse()
        message.server_project_root = Config.project_root
        message.entry_point.components.extend(Config.entry_point.split(os.sep))
        self._send_message(message, 'initialize', address)

    def send_protocol_error(self, error_code, address):
        message = pm.ProtocolError()
        message.error_code = error_code
        self._send_message(message, 'error', address)

    def send_memory_usage_response(
            self, memory_usage, sequence_number, address):
        # Ideally, MessageSender users should not need to know about the INNPV
        # protocol messages. However, to avoid extraneous copies, sometimes
        # callers will pass in constructed messages for sending.
        memory_usage.sequence_number = sequence_number
        self._send_message(memory_usage, 'memory_usage', address)

    def send_analysis_error(self, error_message, sequence_number, address):
        message = pm.AnalysisError()
        message.sequence_number = sequence_number
        message.error_message = error_message
        self._send_message(message, 'analysis_error', address)

    def send_throughput_response(self, throughput, sequence_number, address):
        throughput.sequence_number = sequence_number
        self._send_message(throughput, 'throughput', address)

    def _send_message(self, message, payload_name, address):
        connection = self._connection_manager.get_connection(address)
        enclosing_message = pm.FromServer()
        getattr(enclosing_message, payload_name).CopyFrom(message)
        connection.send_bytes(enclosing_message.SerializeToString())
