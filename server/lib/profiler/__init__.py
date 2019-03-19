import torch

from lib.exceptions import AnalysisError


def to_trainable_model(parse_tree, class_name):
    try:
        executable = compile(parse_tree, '<string>', 'exec')
        scope = {}
        exec(executable, scope, scope)
        model = scope[class_name]().to(torch.device('cuda'))
        model.train()
        return model
    except Exception as ex:
        raise AnalysisError(str(ex), type(ex))
