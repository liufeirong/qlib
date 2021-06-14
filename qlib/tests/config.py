#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

CSI300_MARKET = "csi300"
CSI100_MARKET = "csi100"

CSI300_BENCH = "SH000300"
CSI100_BENCH = "SH000903"

DATASET_ALPHA158_CLASS = "Alpha158"
DATASET_ALPHA360_CLASS = "Alpha360"

###################################
# config
###################################


GBDT_MODEL = {
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
}


RECORD_CONFIG = [
    {
        "class": "SignalRecord",
        "module_path": "qlib.workflow.record_temp",
    },
    {
        "class": "SigAnaRecord",
        "module_path": "qlib.workflow.record_temp",
    },
]


def get_port_ana_config(benchmark):
    return {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.strategy",
            "kwargs": {
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest":  {
            "verbose": False,
            "limit_threshold": 0.095,
            "account": 100000000,
            "benchmark": benchmark,
            "deal_price": "close",
            "open_cost": 0.0015,
            "close_cost": 0.0025,
            "min_cost": 5,
        }
    }


def get_data_handler_config(
    start_time="2008-01-01",
    end_time="2020-08-01",
    fit_start_time="2008-01-01",
    fit_end_time="2014-12-31",
    instruments=CSI300_MARKET,
):
    return {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        "instruments": instruments,
    }


def get_dataset_config(
    dataset_class=DATASET_ALPHA158_CLASS,
    train=("2008-01-01", "2014-12-31"),
    valid=("2015-01-01", "2016-12-31"),
    test=("2017-01-01", "2020-08-01"),
    handler_kwargs={"instruments": CSI300_MARKET},
):
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": dataset_class,
                "module_path": "qlib.contrib.data.handler",
                "kwargs": get_data_handler_config(**handler_kwargs),
            },
            "segments": {
                "train": train,
                "valid": valid,
                "test": test,
            },
        },
    }


def get_gbdt_task(dataset_kwargs={}, handler_kwargs={"instruments": CSI300_MARKET}):
    return {
        "model": GBDT_MODEL,
        "dataset": get_dataset_config(**dataset_kwargs, handler_kwargs=handler_kwargs),
    }


def get_record_lgb_config(dataset_kwargs={}, handler_kwargs={"instruments": CSI300_MARKET}):
    return {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
        },
        "dataset": get_dataset_config(**dataset_kwargs, handler_kwargs=handler_kwargs),
        "record": RECORD_CONFIG,
    }


def get_record_xgboost_config(dataset_kwargs={}, handler_kwargs={"instruments": CSI300_MARKET}):
    return {
        "model": {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
        },
        "dataset": get_dataset_config(**dataset_kwargs, handler_kwargs=handler_kwargs),
        "record": RECORD_CONFIG,
    }


CSI300_DATASET_CONFIG = get_dataset_config(handler_kwargs={"instruments": CSI300_MARKET})
CSI300_GBDT_TASK = get_gbdt_task(handler_kwargs={"instruments": CSI300_MARKET})

CSI100_RECORD_XGBOOST_TASK_CONFIG = get_record_xgboost_config(handler_kwargs={"instruments": CSI100_MARKET})
CSI100_RECORD_LGB_TASK_CONFIG = get_record_lgb_config(handler_kwargs={"instruments": CSI100_MARKET})

# use for rolling_online_managment.py
ROLLING_HANDLER_CONFIG = {
    "start_time": "2014-06-01",
    "end_time": "2021-06-03",
    "fit_start_time": "2014-06-01",
    "fit_end_time": "2020-12-31",
    "instruments": CSI100_MARKET,
}
ROLLING_DATASET_CONFIG = {
    "train": ("2014-06-01", "2019-01-01"),
    "valid": ("2019-01-02", "2020-01-01"),
    "test": ("2020-01-02", "2021-06-03"),
}
CSI100_RECORD_XGBOOST_TASK_CONFIG_ROLLING = get_record_xgboost_config(
    dataset_kwargs=ROLLING_DATASET_CONFIG, handler_kwargs=ROLLING_HANDLER_CONFIG
)
CSI100_RECORD_LGB_TASK_CONFIG_ROLLING = get_record_lgb_config(
    dataset_kwargs=ROLLING_DATASET_CONFIG, handler_kwargs=ROLLING_HANDLER_CONFIG
)

# use for online_management_simulate.py
ONLINE_HANDLER_CONFIG = {
    "start_time": "2018-01-01",
    "end_time": "2021-06-03",
    "fit_start_time": "2018-01-01",
    "fit_end_time": "2018-03-31",
    "instruments": CSI100_MARKET,
}
ONLINE_DATASET_CONFIG = {
    "train": ("2018-01-01", "2018-03-31"),
    "valid": ("2018-04-01", "2018-05-31"),
    "test": ("2018-06-01", "2018-09-10"),
}
CSI100_RECORD_XGBOOST_TASK_CONFIG_ONLINE = get_record_xgboost_config(
    dataset_kwargs=ONLINE_DATASET_CONFIG, handler_kwargs=ONLINE_HANDLER_CONFIG
)
CSI100_RECORD_LGB_TASK_CONFIG_ONLINE = get_record_lgb_config(
    dataset_kwargs=ONLINE_DATASET_CONFIG, handler_kwargs=ONLINE_HANDLER_CONFIG
)

CSI300_PORT_ANA_CONFIG = get_port_ana_config(CSI300_BENCH)
CSI100_PORT_ANA_CONFIG = get_port_ana_config(CSI100_BENCH)
