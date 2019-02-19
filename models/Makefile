PLUGIN_MODELS = ../plugin/lib/models_gen
SERVER_MODELS = ../server/lib/models_gen

.PHONY: models plugin_models server_models

models: plugin_models server_models

plugin_models: messages.proto
	mkdir -p $(PLUGIN_MODELS)
	protoc --js_out=import_style=commonjs,binary:$(PLUGIN_MODELS) $^

server_models: messages.proto
	mkdir -p $(SERVER_MODELS)
	protoc --python_out=$(SERVER_MODELS) $^
