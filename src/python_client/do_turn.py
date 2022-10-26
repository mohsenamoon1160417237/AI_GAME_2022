from base import BaseAgent, Action
from utils import DIAMONDS

import random
import numpy as np


class Phase1:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.prev_diamond = None

    def calc_diamond_scores(self, diamond: str) -> int:
        if self.prev_diamond is None:
            if diamond == DIAMONDS['YELLOW_DIAMOND']:
                return 50
            else:
                return 0
        elif self.prev_diamond == DIAMONDS['YELLOW_DIAMOND']:
            if diamond == DIAMONDS['YELLOW_DIAMOND']:
                return 50
            elif diamond == DIAMONDS['GREEN_DIAMOND']:
                return 200
            elif diamond == DIAMONDS['RED_DIAMOND']:
                return 100
            else:
                return 0
        elif self.prev_diamond == DIAMONDS['GREEN_DIAMOND']:
            if diamond == DIAMONDS['YELLOW_DIAMOND']:
                return 100
            elif diamond == DIAMONDS['GREEN_DIAMOND']:
                return 50
            elif diamond == DIAMONDS['RED_DIAMOND']:
                return 200
            else:
                return 100
        elif self.prev_diamond == DIAMONDS['RED_DIAMOND']:
            if diamond == DIAMONDS['YELLOW_DIAMOND']:
                return 50
            elif diamond == DIAMONDS['GREEN_DIAMOND']:
                return 100
            elif diamond == DIAMONDS['RED_DIAMOND']:
                return 50
            else:
                return 200
        else:
            if diamond == DIAMONDS['YELLOW_DIAMOND']:
                return 250
            elif diamond == DIAMONDS['GREEN_DIAMOND']:
                return 50
            elif diamond == DIAMONDS['RED_DIAMOND']:
                return 100
            else:
                return 50

    def main(self):
        # self.width
        return random.choice(
            [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.DOWN_RIGHT, Action.DOWN_LEFT, Action.UP_LEFT,
             Action.UP_RIGHT, Action.NOOP])
