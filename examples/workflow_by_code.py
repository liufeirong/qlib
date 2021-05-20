#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import qlib
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.tests.data import GetData


if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/qlib_cn_1d"  # target_dir
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    market = "csi300"
    benchmark = "SH000300"

    ###################################
    # train model
    ###################################
    data_handler_config = {
        "start_time": "2015-01-01",
        "end_time": "2021-05-01",
        "fit_start_time": "2015-01-01",
        "fit_end_time": "2020-12-31",
        "instruments": market,
    }

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2015-01-01", "2018-12-31"),
                    "valid": ("2019-01-01", "2020-12-31"),
                    "test": ("2021-01-01", "2020-05-01"),
                },
            },
        },
    }

    # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    example_df = dataset.prepare("train")
    print(example_df.head())

    # start exp
    with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
