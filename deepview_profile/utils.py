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
        context_info_map = analysis['operation']['contextInfoMap']
        if len(context_info_map) > 0:
            filename = list(context_info_map[0]['context']['filePath']['components']).pop()

            try: 
                file_index = encoded_files.index(filename)
            except:
                file_path = os.path.join("", context_info_map[0]['context']['filePath']['components'])

                encoded_file = encode_file("", file_path)
                encoded_files.append(encoded_file)

    return encoded_files

def files_encoded_content(path):
    encoded_files = []

    if os.path.isfile(path):
        return encoded_files

    for root, subFolders, files in os.walk(path):
        for file in files:
            encoded_file = encode_file(root, file)

            if encoded_file is not None:
                encoded_files.append(encoded_file)

    return encoded_files

def encode_file(root, file): 
    file_dict = None
    if os.path.splitext(file)[1] == ".py" and file != "entry_point.py":
        file_dict = {
            "name": file,
            "content": ""
        }

        filename = os.path.join(root, file)

        with open(filename, "r") as f:
            file_content = f.read()
            file_dict["content"] = base64.b64encode(file_content.encode("utf-8")).decode("utf-8")

    return file_dict
