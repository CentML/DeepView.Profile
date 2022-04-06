import logging
import select
import struct
from threading import Thread

from skyline.io.sentinel import Sentinel

logger = logging.getLogger(__name__)


class Connection:
    """
    Manages an open connection to a client.

    This class must be constructed with an already-connected
    socket. Upon receipt of a message on the socket, the
    handler_function will be called with the raw message.

    Socket communication is performed using length-prefixed
    binary protobuf messages.

    The stop function must be called to close the connection.
    """
    def __init__(self, socket, address, handler_function, closed_handler):
        self.address = address
        self._socket = socket
        self._reader = Thread(target=self._socket_read)
        self._handler_function = handler_function
        self._closed_handler = closed_handler
        self._sentinel = Sentinel()

    def start(self):
        self._sentinel.start()
        self._reader.start()

    def stop(self):
        self._sentinel.signal_exit()
        self._reader.join()
        self._socket.close()
        self._sentinel.stop()

    def send_bytes(self, raw_bytes):
        self._socket.sendall(struct.pack('!I', len(raw_bytes)))
        self._socket.sendall(raw_bytes)

    def _socket_read(self):
        buffer = b''
        message_length = -1

        try:
            while True:
                read_ready, _, _ = select.select([
                    self._socket, self._sentinel.read_pipe], [], [])
                if self._sentinel.should_exit(read_ready):
                    logger.debug(
                        "Connection (%s:%d) is being closed.",
                        *self.address,
                    )
                    self._sentinel.consume_exit_signal()
                    break

                data = self._socket.recv(4096)
                if len(data) == 0:
                    logger.debug(
                        "Connection (%s:%d) has been closed by the client.",
                        *self.address,
                    )
                    self._closed_handler(self.address)
                    break

                buffer += data

                # Process all messages that exist in the buffer
                while True:
                    if message_length <= 0:
                        if len(buffer) < 4:
                            break
                        # Network byte order 32-bit unsigned integer
                        message_length = struct.unpack('!I', buffer[:4])[0]
                        buffer = buffer[4:]

                    if len(buffer) < message_length:
                        break

                    try:
                        self._handler_function(
                            buffer[:message_length], self.address)
                    finally:
                        buffer = buffer[message_length:]
                        message_length = -1

        except:
            logger.exception("Connection unexpectedly stopping...")


class ConnectionState:
    def __init__(self):
        # NOTE: This counter is modified by a thread in the main executor, but
        #       will be read by other threads. No R/W lock is needed because of
        #       the Python GIL.
        #
        # NOTE: The sequence number from the client must be non-negative
        self.sequence_number = -1

        # Connections have two states: uninitialized and "ready" (initialized)
        # As a result for simplicity, we use a boolean to represent the state.
        self.initialized = False

        # The plugin may disconnect from us while we are processing a request.
        # We use this flag to indicate whether the connection still "exists"
        # to allow requests to abort early.
        self.connected = True

    def update_sequence(self, request):
        if request.sequence_number <= self.sequence_number:
            return
        self.sequence_number = request.sequence_number

    def is_request_current(self, request):
        return request.sequence_number >= self.sequence_number
