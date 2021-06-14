# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example shows how OnlineManager works with rolling tasks.
There are four parts including first train, routine 1, add strategy and routine 2.
Firstly, the OnlineManager will finish the first training and set trained models to `online` models.
Next, the OnlineManager will finish a routine process, including update online prediction -> prepare tasks -> prepare new models -> prepare signals
Then, we will add some new strategies to the OnlineManager. This will finish first training of new strategies.
Finally, the OnlineManager will finish second routine and update all strategies.
"""

import os
import yaml
import fire
import qlib
from qlib.model.trainer import DelayTrainerR, DelayTrainerRM, TrainerR, TrainerRM, end_task_train, task_train
from qlib.workflow import R
from qlib.workflow.online.strategy import RollingStrategy
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.online.manager import OnlineManager
from qlib.tests.config import CSI100_RECORD_XGBOOST_TASK_CONFIG_ROLLING, CSI100_RECORD_LGB_TASK_CONFIG_ROLLING
from qlib.tests.config import CSI300_PORT_ANA_CONFIG
from qlib.workflow.task.manage import TaskManager
import pandas as pd
pd.options.display.max_columns = None  #列数
pd.options.display.max_rows = 20     #行数
pd.options.display.max_colwidth = None #列宽

def scan_tasks(config_dir="tasks"):
    tasks = []
    for file in os.listdir(config_dir):
        if not file.endswith(".yaml"):
            continue

        with open(os.path.join(config_dir, file)) as fp:
            config = yaml.safe_load(fp)
            tasks.append(config.get("task"))

    return tasks

class RollingOnlineExample:
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/qlib_cn_1d",
        region="cn",
        trainer=DelayTrainerRM(),  # you can choose from TrainerR, TrainerRM, DelayTrainerR, DelayTrainerRM
        task_url="mongodb://localhost:27017/",  # not necessary when using TrainerR or DelayTrainerR
        task_db_name="rolling_db",  # not necessary when using TrainerR or DelayTrainerR
        rolling_step=20,
        tasks=None,
        add_tasks=None,
        port_analysis_config=None,
    ):
        if add_tasks is None:
            add_tasks = scan_tasks("add_tasks")
            # add_tasks = [CSI100_RECORD_LGB_TASK_CONFIG_ROLLING]
        if tasks is None:
            tasks = scan_tasks()
            # tasks = [CSI100_RECORD_XGBOOST_TASK_CONFIG_ROLLING]
        if port_analysis_config is None:
            port_analysis_config = CSI300_PORT_ANA_CONFIG
        mongo_conf = {
            "task_url": task_url,  # your MongoDB url
            "task_db_name": task_db_name,  # database name
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.tasks = tasks
        self.add_tasks = add_tasks
        self.port_analysis_config = port_analysis_config
        self.rolling_step = rolling_step
        strategies = []
        for task in tasks:
            name_id = task["model"]["class"]  # NOTE: Assumption: The model class can specify only one strategy
            strategies.append(
                RollingStrategy(
                    name_id,
                    task,
                    RollingGen(step=rolling_step, rtype=RollingGen.ROLL_SD),
                )
            )
        self.trainer = trainer
        self.rolling_online_manager = OnlineManager(strategies, trainer=self.trainer)

    _ROLLING_MANAGER_PATH = (
        ".RollingOnlineExample"  # the OnlineManager will dump to this file, for it can be loaded when calling routine.
    )

    def worker(self):
        # train tasks by other progress or machines for multiprocessing
        print("========== worker ==========")
        if isinstance(self.trainer, TrainerRM):
            for task in self.tasks + self.add_tasks:
                name_id = task["model"]["class"]
                self.trainer.worker(experiment_name=name_id)
        else:
            print(f"{type(self.trainer)} is not supported for worker.")

    # Reset all things to the first status, be careful to save important data
    def reset(self):
        for task in self.tasks + self.add_tasks:
            name_id = task["model"]["class"]
            TaskManager(task_pool=name_id).remove()
            exp = R.get_exp(experiment_name=name_id)
            for rid in exp.list_recorders():
                exp.delete_recorder(rid)

        if os.path.exists(self._ROLLING_MANAGER_PATH):
            os.remove(self._ROLLING_MANAGER_PATH)

    def first_run(self):
        print("========== reset ==========")
        self.reset()
        print("========== first_run ==========")
        self.rolling_online_manager.first_train()
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def routine(self):
        print("========== load ==========")
        self.rolling_online_manager = OnlineManager.load(self._ROLLING_MANAGER_PATH)
        print("========== routine ==========")
        self.rolling_online_manager.routine()
        print("========== collect results ==========")
        print(self.rolling_online_manager.get_collector()())
        print("========== signals ==========")
        print(self.rolling_online_manager.get_signals())
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def add_strategy(self):
        print("========== load ==========")
        self.rolling_online_manager = OnlineManager.load(self._ROLLING_MANAGER_PATH)
        print("========== add strategy ==========")
        strategies = []
        for task in self.add_tasks:
            name_id = task["model"]["class"]  # NOTE: Assumption: The model class can specify only one strategy
            strategies.append(
                RollingStrategy(
                    name_id,
                    task,
                    RollingGen(step=self.rolling_step, rtype=RollingGen.ROLL_SD),
                )
            )
        self.rolling_online_manager.add_strategy(strategies=strategies)
        print("========== dump ==========")
        self.rolling_online_manager.to_pickle(self._ROLLING_MANAGER_PATH)

    def topk_signals(self, k=10, start_date=None, end_date=None):
        print("========== load ==========")
        self.rolling_online_manager = OnlineManager.load(self._ROLLING_MANAGER_PATH)
        print("========== topk signals ==========")
        signals = self.rolling_online_manager.get_signals()
        signals = signals.groupby(level=0, group_keys=False).nlargest(k)
        print(signals.loc[start_date:end_date, ])

    def get_portfolio_info(self, start_date=None, end_date=None):
        print("========== load ==========")
        self.rolling_online_manager = OnlineManager.load(self._ROLLING_MANAGER_PATH)
        positions_normal, analysis_df = self.rolling_online_manager.get_portfolio(self.port_analysis_config)
        positions_df = pd.DataFrame(positions_normal.values(), index=positions_normal.keys()).stack()
        print("==========backtest analysis========")
        print(analysis_df.loc[start_date:end_date, ])
        print("==========positions ==========")
        print(positions_df.loc[start_date:end_date])

    def main(self):
        self.first_run()
        self.routine()
        self.add_strategy()
        self.routine()


if __name__ == "__main__":
    ####### to train the first version's models, use the command below
    # python rolling_online_management.py first_run

    ####### to update the models and predictions after the trading time, use the command below
    # python rolling_online_management.py routine

    ####### to add new strategy, use the command below
    # python rolling_online_management.py add_strategy

    ####### to define your own parameters, use `--`
    # python rolling_online_management.py first_run --rolling_step=250
    fire.Fire(RollingOnlineExample)
