from base import Action
from utils import GEMS

import random
import numpy as np


class Phase1:
    def __init__(self, Agent):
        self.agent = Agent
        self.height = self.agent.grid_height
        self.width = self.agent.grid_width
        self.map = np.array(self.agent.grid)
        if 'gem_indexes' not in self.agent.__dict__:
            self.agent.gem_indexes = self.make_gem_indexes()
        if 'gem_distances' not in self.agent.__dict__:
            self.agent.gem_distances = self.make_gem_distances()
        if 'prev_gem' not in self.agent.__dict__:
            self.agent.prev_gem = None
        self.scores_env = self.create_score_env()

    def create_score_env(self):
        return np.zeros(dtype=int, shape=(self.height, self.width))

    def calc_gems_scores(self, gem: str) -> int:
        prev_gem = self.agent.prev_gem
        if prev_gem is None:
            if gem == GEMS['YELLOW_GEM']:
                return 50
            else:
                return 0
        elif prev_gem == GEMS['YELLOW_GEM']:
            if gem == GEMS['YELLOW_GEM']:
                return 50
            elif gem == GEMS['GREEN_GEM']:
                return 200
            elif gem == GEMS['RED_GEM']:
                return 100
            else:
                return 0
        elif prev_gem == GEMS['GREEN_GEM']:
            if gem == GEMS['YELLOW_GEM']:
                return 100
            elif gem == GEMS['GREEN_GEM']:
                return 50
            elif gem == GEMS['RED_GEM']:
                return 200
            else:
                return 100
        elif prev_gem == GEMS['RED_GEM']:
            if gem == GEMS['YELLOW_GEM']:
                return 50
            elif gem == GEMS['GREEN_GEM']:
                return 100
            elif gem == GEMS['RED_GEM']:
                return 50
            else:
                return 200
        else:
            if gem == GEMS['YELLOW_GEM']:
                return 250
            elif gem == GEMS['GREEN_GEM']:
                return 50
            elif gem == GEMS['RED_GEM']:
                return 100
            else:
                return 50

    # Will be called only once.
    def make_gem_indexes(self) -> np.array:
        agent_index = np.empty((0, 2), dtype=int)
        gem_indexes = np.empty((0, 3), dtype=int)  # row, col, gem_number

        for row in range(self.map.shape[0]):
            agent = np.where(self.map[row] == 'EA')
            if len(agent[0]) != 0:
                agent_index = np.vstack((agent_index, [row, agent[0][0]]))
                self.agent.agent_index = agent_index
            new_arr = np.where(self.map[row] == '1')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 1]))
            new_arr = np.where(self.map[row] == '2')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 2]))
            new_arr = np.where(self.map[row] == '3')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 3]))
            new_arr = np.where(self.map[row] == '4')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 4]))

        return gem_indexes

    # Will be called only once.
    def make_gem_distances(self) -> np.array:
        distances = np.array([])
        agent_row = self.agent.agent_index[0][0]
        agent_col = self.agent.agent_index[0][1]
        for index in self.agent.gem_indexes:
            row = index[0]
            col = index[1]
            dist_sum = np.sum([np.power((row - agent_row), 2), np.power((col - agent_col), 2)])
            distance = np.sqrt(dist_sum)
            distances = np.append(distances, distance)

        sorted_distances = np.sort(distances)
        return sorted_distances

    def main(self):
        return Action.NOOP
        # return random.choice(
        #     [Action.DOWN, Action.DOWN_RIGHT, Action.DOWN_LEFT, Action.RIGHT, Action.LEFT, Action.UP_LEFT,
        #      Action.UP_RIGHT, Action.UP])
