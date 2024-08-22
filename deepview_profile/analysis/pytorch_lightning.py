from lightning.pytorch.profilers.profiler import Profiler
from typing import Optional, Union
from pathlib import Path
from typing_extensions import override
from lightning.pytorch.core.module import LightningModule
from torch import nn, Tensor

from deepview_profile.analysis.session import AnalysisSession

class DeepviewLightning(Profiler):
    def __init__(self,
                 batch_generator,
                 dirpath: Optional[Union[str, Path]] = None, 
                 filename: Optional[str] = None):
        super().__init__(dirpath=dirpath, filename=filename)
        self.running = False
        self._lightning_module: Optional[LightningModule] = None  # set by ProfilerConnector
        self._batch_generator = batch_generator

    @override
    def start(self, action_name: str) -> None:
        if self._lightning_module is not None:
            if not self.running:
                self.running = True
                inputs, _ = next(iter(self._batch_generator))
                batch_size = inputs.shape[0]

                # def model_provider():
                #     return self._lightning_module()
                
                # def input_provider(batch_size=1):
                #     return next(iter(self._batch_generator))

                # def iterator_provider(model):
                #     def iteration(inputs, targets):
                #         model.training_step(zip(inputs, targets))
                #     return iteration

                # analizer = AnalysisSession(
                #     model_provider=model_provider, 
                #     input_provider=input_provider, 
                #     iteration_provider=iterator_provider,
                #     batch_size=batch_size)
                
                # analizer.measure_throughput()

                self._lightning_module().training_step(next(iter(self._batch_generator)))

    @override
    def stop(self, action_name: str) -> None:
        pass

    @override
    def summary(self) -> str:
        pass

    @override
    def teardown(self, stage: Optional[str]) -> None:
        super().teardown(stage=stage)