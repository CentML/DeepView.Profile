import torch 
import logging
import gc

from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)

def release_memory():
    logger.debug("Emptying cache")
    gc.collect()
    torch.cuda.empty_cache()

def next_message_to_dict(object): 
    message = next(object)
    return MessageToDict(message)
