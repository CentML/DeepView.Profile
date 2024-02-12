import torch
import logging
import gc
import os
import base64
from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)

def release_memory():
    logger.debug("Emptying cache")
    gc.collect()
    torch.cuda.empty_cache()

def next_message_to_dict(object):
    message = next(object)
    return MessageToDict(message)

def files_encoded_unique(operation_tree):
    encoded_files = []

    for analysis in operation_tree:
        context_info_map = analysis["operation"].get("contextInfoMap", None)
        if context_info_map is not None and len(context_info_map) > 0:
            filename = list(
                context_info_map[0]["context"]["filePath"]["components"]
            ).pop()

            already_in_list = next(
                (item for item in encoded_files if item["name"] == filename), None
            )
            if not already_in_list:
                file_path = os.path.join(
                    "", *list(context_info_map[0]["context"]["filePath"]["components"])
                )

                encoded_file = encode_file("", file_path)
                encoded_files.append(encoded_file)

    return encoded_files

def encode_file(root, file):
    file_dict = None
    if os.path.splitext(file)[1] == ".py" and file != "entry_point.py":
        file_dict = {"name": file, "content": ""}

        filename = os.path.join(root, file)

        with open(filename, "r") as f:
            file_content = f.read()
            file_dict["content"] = base64.b64encode(
                file_content.encode("utf-8")
            ).decode("utf-8")

    return file_dict

def model_location_patterns():
    return [
        "./transformers/models[/\w+/]+\w+.py",
        "./transformers/integrations[/\w+/]+\w+.py",
        "./diffusers/models[/\w+/]+\w+.py",
    ]