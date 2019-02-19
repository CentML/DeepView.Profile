import logging

import lib.models_gen.messages_pb2 as m

logger = logging.getLogger(__name__)


class MessageHandler:
    def __init__(self, connection_manager):
        self._connection_manager = connection_manager

    def handle_message(self, raw_data, address):
        try:
            message = m.AnalyzeRequest()
            message.ParseFromString(raw_data)
            logger.info(
                'From (%s:%d) received: %s',
                *address,
                message.source_code,
            )

            response = m.AnalyzeResponse()
            response.response = 'Hello world back!'
            connection = self._connection_manager.get_connection(address)
            connection.write_serialized_message(response.SerializeToString())
        except:
            logger.exception(
                'Processing message from (%s:%d) resulted in an exception.',
                *address,
            )
