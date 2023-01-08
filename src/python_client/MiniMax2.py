from typing import List, Tuple
import numpy as np
from base import Action
from model_based_policy import ModelBasedPolicy
from utils.config import GEMS

class MiniMax(ModelBasedPolicy):
    def __init__(self, agent):

        super().__init__(agent)
        self.agent.agent_index = tuple(self.agent.agent_index)
        if 'wall_indexes' not in self.agent.__dict__:
            self.agent.wall_indexes = self.make_wall_indexes()
        if 'gem_indexes' not in self.agent.__dict__:
            self.agent.gem_indexes = self.make_gem_indexes()
        self.visited_indexes = []
        self.min_iter = 0
        self.max_iter = 0
        if 'gem_groups' not in self.agent.__dict__:  # int
            self.gem_groups = self.group_gems()
        if 'prev_gem' not in self.agent.__dict__:
            if len(np.where(self.map == '1')[0]) > 0 :
                self.agent.prev_gem = None
            else :
                self.agent.prev_gem = '1'
        self.agent.prev_map = self.map
    def make_gem_indexes(self) -> np.array:
        gem_indexes = np.empty((0, 3), dtype=int)  # row, col, gem_number
        for row in range(self.map.shape[0]):
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
    def calc_gems_scores(self, gem: str, prev_gem: str) -> int:
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
    def search_nearby_cells(self, cell: np.array, new_gem_indexes: np.array, gem_group: np.array) -> np.array:
        grid_height = self.height
        grid_width = self.width
        row = cell[0]
        col = cell[1]
        for new_index in new_gem_indexes:
            nearby_cells = np.empty((0, 2))
            if col != grid_height:
                nearby_cells = np.vstack((nearby_cells, [row, col + 1]))
            if col != 0:
                nearby_cells = np.vstack((nearby_cells, [row, col - 1]))
            if row != 0 and col != 0:
                nearby_cells = np.vstack((nearby_cells, [row - 1, col - 1]))
            if row != 0:
                nearby_cells = np.vstack((nearby_cells, [row - 1, col]))
            if row != 0 and col != grid_height:
                nearby_cells = np.vstack((nearby_cells, [row - 1, col + 1]))
            if row != grid_width and col != 0:
                nearby_cells = np.vstack((nearby_cells, [row + 1, col - 1]))
            if row != grid_width:
                nearby_cells = np.vstack((nearby_cells, [row + 1, col]))
            if row != grid_width and col != grid_height:
                nearby_cells = np.vstack((nearby_cells, [row + 1, col + 1]))

            reshaped_nearby_cells = [x.tolist() for x in nearby_cells]
            for num, cl in enumerate(reshaped_nearby_cells):
                cl.append(new_index[2])
                reshaped_nearby_cells[num] = [int(x) for x in cl]
            if new_index.tolist() in reshaped_nearby_cells:
                gem_group = np.vstack((gem_group, new_index.astype(int)))

        return gem_group
    def search_gems(self, index: np.array, gem_group: np.array, searched_gems: np.array, gem_groups: list,
                    gem_indexes: np.array) -> list:
        new_gem_indexes = np.array(
            [x for x in gem_indexes.tolist() if x not in searched_gems.tolist() and x not in gem_group.tolist()])
        gem_group = self.search_nearby_cells(index, new_gem_indexes, gem_group)
        searched_gems = np.vstack((searched_gems, index))

        not_searched_gems = [
            x for x in gem_group.tolist() if x not in searched_gems.tolist()]
        if len(not_searched_gems) == 0:
            gem_groups.append(gem_group)
            return gem_groups
        index = not_searched_gems[0]
        gem_groups = self.search_gems(
            index, gem_group, searched_gems, gem_groups, gem_indexes)
        if gem_groups is not None:
            return gem_groups
    def group_gems(self) -> list:
        gem_groups = []
        gem_indexes = self.agent.gem_indexes
        while True:
            gem_group = np.empty((0, 3))
            searched_gems = np.empty((0, 3), dtype=int)
            if len(gem_groups) == 0:
                index = gem_indexes[0]
            else:
                flatten_groups = [x.tolist() for x in gem_groups]
                index_ls = [x for x in gem_indexes.tolist() if x not in [
                    m for y in flatten_groups for m in y]]
                if len(index_ls) == 0:
                    return gem_groups
                index = index_ls[0]
            gem_group = np.vstack((gem_group, index))
            gem_groups = self.search_gems(
                index, gem_group, searched_gems, gem_groups, gem_indexes)
    def arrange_gem(self, gem_group, best_arrangement_of_gem, prev_gem: str) -> list:
        # gem_group is first arrange of gem
        # best_arrangement_of_gem is best arrange of gem
        # best_arrangement_of_gem is list that first element is total score and second element
        if len(gem_group) == 0:
            return best_arrangement_of_gem

        list = []
        # list of tuple :
        # first item in tuple is color of gem and second item is score of gem
        if 1 in gem_group:
            list.append(("1", self.calc_gems_scores('1', prev_gem)))
        if 2 in gem_group:
            list.append(("2", self.calc_gems_scores('2', prev_gem)))
        if 3 in gem_group:
            list.append(("3", self.calc_gems_scores('3', prev_gem)))
        if 4 in gem_group:
            list.append(("4", self.calc_gems_scores('4', prev_gem)))
        list.sort(key=lambda a: a[1], reverse=True)

        prev_gem = str(list[0][0])
        best_arrangement_of_gem[0] += list[0][1]
        best_arrangement_of_gem.append(prev_gem)
        gem_group.remove(int(prev_gem))

        best_arrangement_of_gem = self.arrange_gem(
            gem_group, best_arrangement_of_gem, prev_gem)
        if best_arrangement_of_gem is not None:
            return best_arrangement_of_gem
    def calc_prev_gem(self):
        gems = ["1", "2", "3", "4"]
        x_agent, y_agent = self.agent.agent_index
        current_cell = self.agent.prev_map[x_agent][y_agent]
        if current_cell in gems:
            self.agent.prev_gem = current_cell
    def find_best_area(self) :
        prev_gem = self.agent.prev_gem
        for group in self.gem_groups:

            # find best arrange for this gem group

            best_arrange = self.arrange_gem(
                group[:, 2].tolist(), [self.agent.agent_scores[0]], prev_gem)
            first_gem_of_arrange = best_arrange[1]

            # check if we have same gem in the first arrange

            index_of_first_gem_of_arrange = ()
            for i in range(0, group.shape[0]):
                if int(group[i][2]) == int(first_gem_of_arrange):
                    index_of_first_gem_of_arrange = (group[i][0], group[i][1])
                    break

        return index_of_first_gem_of_arrange

    def calc_aim(self):

        index_of_first_gem = self.find_best_area()
        return index_of_first_gem


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
            if index != self.agent.agent_index and index not in self.agent.wall_indexes and index not in self.visited_indexes:
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
    def get_action_from_state (self , state):
        (i,j) = state
        i_agent , j_agent = self.agent.agent_index
        if ( i_agent + 1  == i ) :
            if (j_agent == j):
                return 'DOWN'
            if (j_agent + 1  == j):
                return 'DOWN_RIGHT'
            if (j_agent -1 == j):
                return 'DOWN_LEFT'
        if (i_agent - 1 == i) :
            if (j_agent == j):
                return 'UP'
            if (j_agent + 1  == j):
                return 'UP_RIGHT'
            if (j_agent -1 == j):
                return 'UP_LEFT'
        if (i_agent == i) :
            if (j_agent + 1 == j):
                return 'RIGHT'
            if (j_agent -1  == j):
                return 'LEFT'
        return 'NOOP'

    def perform_action(self, action: str):
        if action == 'UP':
            return Action.UP

        elif action == 'DOWN':
            return Action.DOWN

        elif action == 'LEFT':
            return Action.LEFT

        elif action == 'RIGHT':
            return Action.RIGHT

        elif action == 'DOWN_RIGHT':
            return Action.DOWN_RIGHT

        elif action == 'DOWN_LEFT':
            return Action.DOWN_LEFT

        elif action == 'UP_LEFT':
            return Action.UP_LEFT

        elif action == 'UP_RIGHT':
            return Action.UP_RIGHT
        elif action == 'NOOP':
            return Action.NOOP
    def main(self):
        print("map :",self.map)

        index_of_first_gem = self.calc_aim()
        # print(index_of_first_gem)
        print("agent : ",self.agent.agent_index)
        self.goal_index = index_of_first_gem
        print("goal " ,self.goal_index)
        scores = self.minimax(self.agent.agent_index, True, [])
        print(scores)
        next_state = scores[1][0]
        next_action = self.get_action_from_state(next_state)
        return self.perform_action(next_action)
        # return Action.UP
        