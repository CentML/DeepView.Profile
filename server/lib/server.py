import logging
from concurrent.futures import ThreadPoolExecutor

from lib.io.connection_acceptor import ConnectionAcceptor
from lib.io.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


class INNPVServer:
    def __init__(self, host, port):
        self._requested_host = host
        # This is the port the user specified on the command line (it can be 0)
        self._requested_port = port
        self._connection_acceptor = ConnectionAcceptor(
            self._requested_host,
            self._requested_port,
            self._on_new_connection,
        )
        self._connection_manager = ConnectionManager(
            self._on_message,
            self._on_connection_closed,
        )
        self._main_executor = ThreadPoolExecutor(max_workers=1)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self):
        self._connection_acceptor.start()
        logger.info("INNPV server has started.")

    def stop(self):
        def shutdown():
            self._connection_acceptor.stop()
            self._connection_manager.stop()
        self._main_executor.submit(shutdown).result()
        self._main_executor.shutdown()
        logger.info("INNPV server has shut down.")

    def _on_message(self, data, address):
        # Do not call directly - called by a connection
        self._main_executor.submit(
            self._handle_message,
            data,
            address,
        )

    def _on_new_connection(self, socket, address):
        # Do not call directly - called by _connection_acceptor
        self._main_executor.submit(
            self._connection_manager.register_connection,
            socket,
            address,
        )

    def _on_connection_closed(self, address):
        # Do not call directly - called by a connection when it is closed
        self._main_executor.submit(
            self._connection_manager.remove_connection,
            address,
        )

    def _handle_message(self, data, address):
        try:
            logger.debug("From (%s:%d) received message: %s", *address, data)
            connection = self._connection_manager.get_connection(address)
            connection.write_string_message("Message received!")
        except:
            logger.exception("Exception occurred when handling message.")
