GEN_DIR = protocol_gen

PLUGIN_GEN = ../plugin/lib/$(GEN_DIR)
PLUGIN_GEN_FILE = $(PLUGIN_GEN)/innpv_pb.js

SERVER_GEN = ../skyline/$(GEN_DIR)
SERVER_GEN_FILE = $(SERVER_GEN)/innpv_pb2.py

PROTO_FILE = innpv.proto

.PHONY: all clean

all: $(PLUGIN_GEN_FILE) $(SERVER_GEN_FILE)

$(PLUGIN_GEN_FILE): $(PROTO_FILE)
	mkdir -p $(PLUGIN_GEN)
	protoc --js_out=import_style=commonjs,binary:$(PLUGIN_GEN) $^

$(SERVER_GEN_FILE): $(PROTO_FILE)
	mkdir -p $(SERVER_GEN)
	protoc --python_out=$(SERVER_GEN) $^

clean:
	rm -rf $(PLUGIN_GEN) $(SERVER_GEN)
