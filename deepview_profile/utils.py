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

def files_encoded_content(path):
    encoded_files = []

    if not os.path.isfile(path):

        for root, subFolders, files in os.walk(path):
            for file in files:
                if os.path.splitext(file)[1] == ".py" and file != "entry_point.py":
                    file_dict = {
                        "name": file,
                        "content": ""
                    }

                    filename = os.path.join(root, file)

                    with open(filename, "r") as f:
                        file_content = f.read()
                        file_dict["content"] = base64.b64encode(file_content.encode("utf-8")).decode("utf-8")
                        encoded_files.append(file_dict)

    return encoded_files

