from typing import List, Tuple
import numpy as np

from model_based_policy import ModelBasedPolicy


class MiniMax(ModelBasedPolicy):
    def __init__(self, agent, goal_index: tuple):
        super().__init__(agent)
        if 'wall_indexes' not in self.agent.__dict__:
            self.agent.wall_indexes = self.make_wall_indexes()
        if 'gem_indexes' not in self.agent.__dict__:
            self.agent.gem_indexes = None
        self.goal_index = goal_index
        self.visited_indexes = []
        self.agent_index = tuple(self.agent.agent_index)
        self.min_iter = 0
        self.max_iter = 0

    def make_wall_indexes(self) -> np.array:
        wall_indexes = []
        for row in range(self.height):
            for col in range(self.width):
                if self.map[row][col] == "W":
                    wall_indexes.append((row, col))
        return wall_indexes

    def make_nearby_indexes(self, cur_index, indexes: List):
        near_by_indexes = []
        i, j = cur_index
        if i != 0:
            near_by_indexes.append((i - 1, j))
        if i != 0 and j != 0:
            near_by_indexes.append((i - 1, j - 1))
        if i != 0 and j != self.width - 1:
            near_by_indexes.append((i - 1, j + 1))
        if j != 0:
            near_by_indexes.append((i, j - 1))
        if i != self.height - 1 and j != 0:
            near_by_indexes.append((i + 1, j - 1))
        if i != self.height - 1:
            near_by_indexes.append((i + 1, j))
        if i != self.height - 1 and j != self.width - 1:
            near_by_indexes.append((i + 1, j + 1))
        if j != self.width - 1:
            near_by_indexes.append((i, j + 1))

        new_near_by_indexes = []
        for index in near_by_indexes:
            if index != self.agent_index and index not in self.agent.wall_indexes and index not in self.visited_indexes:
                new_indexes = indexes.copy()
                new_indexes.append(index)
                new_near_by_indexes.append((index, new_indexes))

        return new_near_by_indexes

    def minimax(self, cur_index: tuple, max_turn: bool, indexes: list) -> list:
        """"
        Main function
        """
        if cur_index == self.goal_index:
            return [self.heuristic(cur_index), indexes[0]]

        near_by_indexes = self.make_nearby_indexes(cur_index, indexes)
        self.visited_indexes.append(cur_index)
        if len(near_by_indexes) == 0:
            # return [self.heuristic(cur_index), indexes[0]]
            return [0, indexes[0]]

        if max_turn:
            self.max_iter += 1
            results = [self.minimax(index[0], False, index[1]) for index in near_by_indexes]
            max_reward = 0
            reward_index = 0
            for enum, itm in enumerate(results):
                if itm[0] > max_reward:
                    max_reward = itm[0]
                    reward_index = enum
            return [max_reward, near_by_indexes[reward_index][1]]

        else:
            self.min_iter += 1
            results = [self.minimax(index[0], False, index[1]) for index in near_by_indexes]
            min_reward = 999999
            reward_index = 0
            for enum, itm in enumerate(results):
                if itm[0] < min_reward:
                    min_reward = itm[0]
                    reward_index = enum
            return [min_reward, near_by_indexes[reward_index][1]]

    def heuristic(self, cur_index: tuple) -> int:
        """
        Calculates score of the current index. And if current index is equal to the self.goal_index(gem index) it should
        return the gem score.
        """
        return 10

    def main(self):
        scores = self.minimax(self.agent_index, True, [])
        print(scores)
        return scores
