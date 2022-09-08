
import argparse
import json
import os
import socket
import struct
import subprocess
import threading
from termcolor import colored

from skyline.protocol_gen import innpv_pb2

def stream_monitor(stream, callback=None):
    for line in stream: callback(line)

def socket_monitor(socket, callback=None):
    try:
        while True:
            msg_len = struct.unpack(">I", socket.recv(4))[0]
            msg = socket.recv(msg_len)
            print(f"Received message of length {msg_len}")
            callback(msg)
    except OSError as e:
        print("Socket is closed. Terminating thread.")

class BackendContext:
    def __init__(self, skyline_bin, entry_point):
        self.process = None
        self.skyline_bin = skyline_bin
        self.entry_point = entry_point
        self.state = 0

    def on_message_stdout(self, message):
        message = message.decode("ascii").rstrip()

    def on_message_stderr(self, message):
        message = message.decode("ascii").rstrip()
        print("stderr", message)
        if "Skyline interactive profiling session started!" in message:
            self.state = 1

    def spawn_process(self):
        # Skyline expects the entry_point filename to be relative
        working_dir = os.path.dirname(self.entry_point)
        entry_filename = os.path.basename(self.entry_point)
        launch_command = [self.skyline_bin, "interactive", "--skip-atom", entry_filename]

        # Launch backend + listener threads for stdout and stderr
        self.process = subprocess.Popen(launch_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_dir)
        self.stdout_thread = threading.Thread(target=stream_monitor, args=(self.process.stdout, self.on_message_stdout))
        self.stdout_thread.start()
        self.stderr_thread = threading.Thread(target=stream_monitor, args=(self.process.stderr, self.on_message_stderr))
        self.stderr_thread.start()

    def join(self):
        self.process.wait()
        
class SkylineSession:
    def __init__(self):
        self.seq_num = 0
        self.received_messages = []

    def connect(self, addr, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((addr, port))

        self.listener_thread = threading.Thread(target=socket_monitor, args=(self.socket, self.handle_message))
        self.listener_thread.start()

    def send_message(self, message):
        msg = innpv_pb2.FromClient()
        msg.sequence_number = self.seq_num
        self.seq_num += 1

        if isinstance(message, innpv_pb2.InitializeRequest):
            msg.initialize.CopyFrom(message)
        elif isinstance(message, innpv_pb2.AnalysisRequest):
            msg.analysis.CopyFrom(message)

        buf = msg.SerializeToString()
        length_buffer = struct.pack(">I", len(buf))

        self.socket.sendall(length_buffer)
        self.socket.sendall(buf)

    def send_initialize_request(self):
        request = innpv_pb2.InitializeRequest()
        request.protocol_version = 5
        self.send_message(request)

    def send_analysis_request(self):
        request = innpv_pb2.AnalysisRequest()
        request.mock_response = False
        self.send_message(request)
    
    def handle_message(self, message):
        from_server = innpv_pb2.FromServer()
        from_server.ParseFromString(message)
        print("From Server:")
        print(from_server)

        self.received_messages.append(from_server)
        print("received messages:", len(self.received_messages))

    def cleanup(self):
        # Closing the socket should cause the listener thread to die
        self.socket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = json.load(fp)

    context = BackendContext(config["skyline_bin"], config["entry_point"])
    context.spawn_process()

    sess = SkylineSession()
    while context.state == 0: pass
    print("Backend ready, connecting")
    sess.connect("localhost", 60120)
    sess.send_initialize_request()
    sess.send_analysis_request()

    context.join()
