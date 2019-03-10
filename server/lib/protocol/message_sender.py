from random import random

import lib.models_gen.messages_pb2 as m


class MessageSender:
    def __init__(self, connection_manager):
        self._connection_manager = connection_manager

    def send_analyze_response(self, batch_size, model_operations, address):
        message = m.AnalyzeResponse()
        message.batch_size = batch_size

        for operation in model_operations:
            pb = message.results.add()
            operation.fill_protobuf(pb)
            # Fake the runtime - use a random value between 100 us and 200 us
            pb.runtime_us = random() * 100 + 100

        # Use mock data for the throughput & memory
        throughput_info = m.ThroughputInfo()
        throughput_info.throughput = 1337
        throughput_info.max_throughput = 2000
        throughput_info.throughput_limit = 1890
        message.throughput.CopyFrom(throughput_info)

        memory_info = m.MemoryInfo()
        memory_info.usage = 2048
        memory_info.max_capacity = 8192
        message.memory.CopyFrom(memory_info)

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
