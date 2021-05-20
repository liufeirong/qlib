# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

try:
    from .double_ensemble import DEnsembleModel
    from .gbdt import LGBModel
except ModuleNotFoundError:
    DEnsembleModel, LGBModel = None, None
    print("Please install necessary libs for DEnsembleModel and LGBModel, such as lightgbm.")

pytorch_classes = ()

all_model_classes = (DEnsembleModel, LGBModel) + pytorch_classes
