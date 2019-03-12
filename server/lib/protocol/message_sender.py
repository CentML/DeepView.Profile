from random import random

import lib.models_gen.messages_pb2 as m


class MessageSender:
    def __init__(self, connection_manager):
        self._connection_manager = connection_manager

    def send_analyze_response(self, annotation_info, model_operations, address):
        message = m.AnalyzeResponse()
        message.batch_size = annotation_info.input_size[0]
        message.innpv_annotation.line = annotation_info.line
        message.innpv_annotation.column = annotation_info.column

        for operation in model_operations:
            pb = message.results.add()
            operation.fill_protobuf(pb)
            # Fake the runtime - use a random value between 100 us and 200 us
            pb.runtime_us = random() * 100 + 100

        # Use mock data for the throughput & memory
        message.throughput.throughput = 1000
        message.throughput.max_throughput = 1250
        message.throughput.throughput_limit = 1250
        message.throughput.runtime_model_ms.coefficient = 0.80320687
        message.throughput.runtime_model_ms.bias = 9.16780518

        message.memory.usage_mb = 1828
        message.memory.max_capacity_mb = 8192
        message.memory.usage_model_mb.coefficient = 10.8583003
        message.memory.usage_model_mb.bias = 1132.56299

        self._send_message(message, 'analyze_response', address)

    def send_analyze_error(self, error_message, address):
        message = m.AnalyzeError()
        message.error_message = error_message
        self._send_message(message, 'analyze_error', address)

    def _send_message(self, message, payload_name, address):
        connection = self._connection_manager.get_connection(address)
        enclosing_message = m.ServerMessage()
        getattr(enclosing_message, payload_name).CopyFrom(message)
        connection.send_bytes(enclosing_message.SerializeToString())
