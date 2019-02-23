import lib.models_gen.messages_pb2 as m


class MessageSender:
    def __init__(self, connection_manager):
        self._connection_manager = connection_manager

    def send_analyze_response(self, response, address):
        message = m.AnalyzeResponse()
        message.response = response
        self._send_message(message, 'analyze_response', address)

    def _send_message(self, message, payload_name, address):
        connection = self._connection_manager.get_connection(address)
        enclosing_message = m.ServerMessage()
        getattr(enclosing_message, payload_name).CopyFrom(message)
        connection.send_bytes(enclosing_message.SerializeToString())
