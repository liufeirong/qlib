# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

try:
    from .double_ensemble import DEnsembleModel
    from .gbdt import LGBModel
except ModuleNotFoundError:
    DEnsembleModel, LGBModel = None, None
    print("Please install necessary libs for DEnsembleModel and LGBModel, such as lightgbm.")

# import pytorch models
try:
    from .pytorch_nn import DNNModelPytorch
    from .pytorch_gru import GRU
    pytorch_classes = (DNNModelPytorch, GRU)
except ModuleNotFoundError:
    pytorch_classes = ()
    print("Please install necessary libs for PyTorch models.")

# import tensorflow models
try:
    from .tensorflow_aemlp import AutoencoderMlpModel
    tensorflow_classes = (AutoencoderMlpModel,)
except ModuleNotFoundError:
    tensorflow_classes = ()
    print("Please install necessary libs for AutoencoderMlpModel, such as tensorflow.")


all_model_classes = (DEnsembleModel, LGBModel) + tensorflow_classes + pytorch_classes
