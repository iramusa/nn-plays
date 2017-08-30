#!/usr/bin/env python3

import datetime

from balls_sim import DEFAULT_CONFIG
from hydranet import HydraNet, DEFAULT_SCHEME


class Experiment(object):
    def __init__(self, ctrl_var):
        self.date = datetime.datetime.now().strftime('%y-%m-%d_%H:%M')
        self.ctrl_var = ctrl_var

