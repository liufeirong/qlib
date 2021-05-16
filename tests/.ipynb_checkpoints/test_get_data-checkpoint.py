#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
import shutil
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("scripts")))
from get_data import GetData

import qlib
from qlib.data import D

DATA_DIR = Path(__file__).parent.joinpath("test_get_data")
SOURCE_DIR = DATA_DIR.joinpath("source")
SOURCE_DIR.mkdir(exist_ok=True, parents=True)
QLIB_DIR = DATA_DIR.joinpath("qlib")
QLIB_DIR.mkdir(exist_ok=True, parents=True)

