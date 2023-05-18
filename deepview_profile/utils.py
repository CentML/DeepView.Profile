import torch 
import logging
import gc

logger = logging.getLogger(__name__)

def release_memory():
    logger.debug("Emptying cache")
    gc.collect()
    torch.cuda.empty_cache()