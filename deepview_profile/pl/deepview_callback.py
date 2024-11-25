from typing import Callable, Tuple

import time
import os
import json
import torch
import sys

try:
    import pytorch_lightning as pl
except ImportError:
    sys.exit("Please install pytorch-lightning:\nuse: pip install lightning\nExiting...")

from termcolor import colored
from deepview_profile.pl.deepview_interface import trigger_profiling


class DeepViewProfilerCallback(pl.Callback):
    def __init__(self, profile_name: str):
        super().__init__()
        self.profiling_triggered = False
        self.output_filename = f"{profile_name}_{int(time.time())}.json"

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
    ):

        # only do this once
        if self.profiling_triggered:
            return

        print(colored("DeepViewProfiler: Running profiling.", "green"))

        """
        need 3 things:

            input_provider: just return batch
            model_provider: just return pl_module
            iteration_provider: a lambda function that (a) calls pl_module.forward_step and (b) calls loss.backward
        """
        initial_batch_size = batch[0].shape[0]

        def input_provider(batch_size: int = initial_batch_size) -> Tuple:
            model_inputs = list()
            for elem in batch:
                # we assume the first dimension is the batch dimension
                model_inputs.append(
                    elem[:1].repeat([batch_size] + [1 for _ in elem.shape[1:]])
                )
            return (tuple(model_inputs), 0)

        model_provider = lambda: pl_module

        def iteration_provider(module: torch.nn.Module) -> Callable:
            def iteration(*args, **kwargs):
                loss = module.training_step(*args, **kwargs)
                loss.backward()

            return iteration

        project_root = os.getcwd()

        output = trigger_profiling(
            project_root,
            "entry_point.py",
            initial_batch_size,
            input_provider,
            model_provider,
            iteration_provider,
        )

        with open(self.output_filename, "w") as fp:
            json.dump(output, fp, indent=4)

        print(
            colored(
                f"DeepViewProfiler: Profiling complete! Report written to ", "green"
            )
            + colored(self.output_filename, "green", attrs=["bold"])
        )
        print(
            colored(
                f"DeepViewProfiler: View your report at https://deepview.centml.ai",
                "green",
            )
        )
        self.profiling_triggered = True
