import logging
import socket

from skyline.io.connection import Connection, ConnectionState
from skyline.exceptions import NoConnectionError

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self, message_handler, closed_handler):
        self._connections = {}
        self._message_handler = message_handler
        self._closed_handler = closed_handler

    def register_connection(self, opened_socket, address):
        self._connections[address] = (
            Connection(
                opened_socket,
                address,
                self._message_handler,
                self._closed_handler,
            ),
            ConnectionState(),
        )
        self._connections[address][0].start()

    def remove_connection(self, address):
        connection, state = self.get_connection_tuple(address)
        connection.stop()
        state.connected = False
        del self._connections[address]
        logger.debug("Removed connection to (%s:%d).", *address)

    def get_connection(self, address):
        return self.get_connection_tuple(address)[0]

    def get_connection_state(self, address):
        return self.get_connection_tuple(address)[1]

    def get_connection_tuple(self, address):
        if address not in self._connections:
            host, port = address
            raise NoConnectionError(
                "Connection to ({}:{}) does not exist.".format(host, port))
        return self._connections[address]

    def broadcast(self, string_message):
        for _, (connection, _) in self._connections.items():
            connection.write_string_message(string_message)

    def connect_to(self, host, port):
        new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        new_socket.connect((host, port))
        self.register_connection(new_socket, (host, port))

    def stop(self):
        for _, (connection, state) in self._connections.items():
            connection.stop()
            state.connected = False
        self._connections.clear()
