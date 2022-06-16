import inspect
import torch.optim as optim


############################################################
OPTIMIZERS_ = optim.__dict__
OPTIMIZERS = {o: OPTIMIZERS_[o] for o in OPTIMIZERS_ if not o.startswith("_") and inspect.isclass(OPTIMIZERS_[o])}
############################################################


############################################################
OPTIMIZERS_PARAMS = {k: {} for k in OPTIMIZERS.keys()}

OPTIMIZERS_PARAMS["SGD"] = dict(momentum=0.9)
OPTIMIZERS_PARAMS["Adam"] = dict(betas=(0.9, 0.999))
############################################################
