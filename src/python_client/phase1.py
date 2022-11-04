import numpy as np

from base import Action
from utils.config import GEMS
from find_path import FindPath


class Phase1:
    def __init__(self, Agent):
        self.agent = Agent
        self.height = self.agent.grid_height
        self.width = self.agent.grid_width
        self.map = np.array(self.agent.grid)
        if 'gem_indexes' not in self.agent.__dict__:  # int
            self.agent.gem_indexes = self.make_gem_indexes()
        if 'wall_indexes' not in self.agent.__dict__:  # int
            self.agent.wall_indexes = self.make_wall_indexes()
        if 'key_indexes' not in self.agent.__dict__:  # str
            self.agent.key_indexes = self.make_key_indexes()
        if 'door_indexes' not in self.agent.__dict__:  # str
            self.agent.door_indexes = self.make_door_indexes()
        if 'barbed_indexes' not in self.agent.__dict__:  # int
            self.agent.barbed_indexes = self.make_barbed_indexes()
        if 'gem_groups' not in self.agent.__dict__:  # int
            self.agent.gem_groups = self.group_gems()
        if 'prev_gem' not in self.agent.__dict__:
            self.agent.prev_gem = None
        if 'green_key_number' not in self.agent.__dict__:
            self.agent.green_key_number = 0
        if 'yellow_key_number' not in self.agent.__dict__:
            self.agent.yellow_key_number = 0
        if 'red_key_number' not in self.agent.__dict__:
            self.agent.red_key_number = 0

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

    def find_best_area(self) -> list:
        list = []
        prev_gem = self.agent.prev_gem
        for group in self.agent.gem_groups:

            # find best arrange for this gem group

            best_arrange = self.arrange_gem(
                group[:, 2].tolist(), [0], prev_gem)
            first_gem_of_arrange = best_arrange[1]
            index_of_first_gem_of_arrange = ()
            for i in range(0, group.shape[0]):
                if int(group[i][2]) == int(first_gem_of_arrange):
                    index_of_first_gem_of_arrange = (group[i][0], group[i][1])

            # cal distance from agent to first gem of this gem group
            cost = self.find_path_for_gem_group(
                group, index_of_first_gem_of_arrange, self.agent.agent_index[0])

            cost_and_score = cost + best_arrange[0]

            list.append(
                (cost_and_score, best_arrange[1:], index_of_first_gem_of_arrange))
        list.sort(key=lambda a: a[0], reverse=True)
        self.agent.gems_arrangement = list
        # print(list)
        return list[0]

    def calc_aim(self):
        # score_of_area = self.find_best_area()[0]
        # arrange_of_area = self.find_best_area()[1]
        # index_of_first_gem = self.find_best_area()[2]
        # print(index_of_first_gem)
        score_of_area, arrange_of_area, index_of_first_gem = self.find_best_area()
        return index_of_first_gem

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
    def make_wall_indexes(self) -> np.array:
        wall_indexes = np.empty((0, 2), dtype=int)  # row, col
        for row in range(self.map.shape[0]):
            new_arr = np.where(self.map[row] == 'W')
            for col in new_arr[0]:
                wall_indexes = np.vstack((wall_indexes, [row, col]))
        return wall_indexes

    # Will be called only once.
    def make_key_indexes(self) -> np.array:
        key_indexes = np.empty((0, 3), dtype=str)  # row, col, key_color

        for row in range(self.map.shape[0]):
            new_arr = np.where(self.map[row] == 'r')
            for col in new_arr[0]:
                key_indexes = np.vstack((key_indexes, [row, col, 'r']))
            new_arr = np.where(self.map[row] == 'g')
            for col in new_arr[0]:
                key_indexes = np.vstack((key_indexes, [row, col, 'g']))
            new_arr = np.where(self.map[row] == 'y')
            for col in new_arr[0]:
                key_indexes = np.vstack((key_indexes, [row, col, 'y']))

        return key_indexes

    # Will be called only once.
    def make_door_indexes(self) -> np.array:
        door_indexes = np.empty((0, 3), dtype=str)  # row, col, door_color

        for row in range(self.map.shape[0]):
            new_arr = np.where(self.map[row] == 'R')
            for col in new_arr[0]:
                door_indexes = np.vstack((door_indexes, [row, col, 'R']))
            new_arr = np.where(self.map[row] == 'G')
            for col in new_arr[0]:
                door_indexes = np.vstack((door_indexes, [row, col, 'G']))
            new_arr = np.where(self.map[row] == 'Y')
            for col in new_arr[0]:
                door_indexes = np.vstack((door_indexes, [row, col, 'Y']))

        return door_indexes

    # Will be called only once.
    def make_barbed_indexes(self) -> np.array:
        barbed_indexes = np.empty((0, 2), dtype=int)  # row, col
        for row in range(self.map.shape[0]):
            new_arr = np.where(self.map[row] == '*')
            for col in new_arr[0]:
                barbed_indexes = np.vstack((barbed_indexes, [row, col]))
        return barbed_indexes

    def calc_gem_group_distances(self) -> np.array:
        distances = np.array([])
        agent_row = self.agent.agent_index[0][0]
        agent_col = self.agent.agent_index[0][1]
        for group in self.agent.gems_arrangement:
            gem_row, gem_col = group[2]
            dist_sum = np.sum(
                [np.power((gem_row - agent_row), 2), np.power((gem_col - agent_col), 2)])
            distance = np.sqrt(dist_sum)
            distances = np.append(distances, distance)

        return distances

    # calc neighbor is done except color home
    def calc_neighbors(self, i_agent, j_agent) -> np.array:
        # calculate cost from initial state to next state
        # A* function : h(n) = f(n) + g(n)
        # this method is function f(n) in heuristic
        # print("yess")
        cost = np.zeros((3, 3), dtype=int)
        x = 0
        item_type = ''
        item_index = ()
        aim_index = self.calc_aim()
        best_gem_group = ''
        neighbors = np.empty((0, 3), dtype=int)
        for i in range(i_agent - 1, i_agent + 2):
            y = 0
            for j in range(j_agent - 1, j_agent + 2):
                if i == i_agent and j == j_agent:
                    cost[x][y] += 0

                elif i != -1 and j != -1 and j != self.width and i != self.height:
                    if self.map[i][j] == 'E':
                        item_type = "empty"
                        item_index = (x, y)

                        if (i == i_agent):
                            cost[x][y] += -1
                        elif (j == j_agent):
                            cost[x][y] += -1
                        else:
                            cost[x][y] += -2

                        cost[x][y] += self.find_path_for_gem_group(
                            best_gem_group, aim_index, np.array([x, y]))
                        neighbors = np.vstack(
                            (neighbors, [cost[x][y], item_type, item_index]))

                    elif self.map[i][j] == 'W':
                        cost[x][y] += -10000

                    elif self.map[i][j] == 'g' or self.map[i][j] == 'r' or self.map[i][j] == 'y':

                        # its key
                        if self.map[i][j] == 'g':
                            item_type = "get_green_key"

                        elif self.map[i][j] == 'r':
                            item_type = "get_red_key"

                        elif self.map[i][j] == 'y':
                            item_type = "get_yellow_key"

                        cost[x][y] += 10
                        cost[x][y] += self.find_path_for_gem_group(
                            best_gem_group, aim_index, np.array([x, y]))
                        item_index = (x, y)
                        neighbors = np.vstack(
                            (neighbors, [cost[x][y], item_type, item_index]))

                    elif self.map[i][j] == '*':
                        item_type = "barbed"
                        item_index = (x, y)
                        cost[x][y] += -20
                        cost[x][y] += self.find_path_for_gem_group(
                            best_gem_group, aim_index, np.array([x, y]))
                        neighbors = np.vstack(
                            (neighbors, [cost[x][y], item_type, item_index]))

                    elif self.map[i][j] == 'G' or self.map[i][j] == 'R' or self.map[i][j] == 'Y':
                        # its lock
                        # if we have key :
                        cost[x][y] += 10
                        cost[x][y] += self.find_path_for_gem_group(
                            best_gem_group, aim_index, np.array([x, y]))
                        item_index = (x, y)

                        if self.map[i][j] == 'G':
                            item_type = "unlocked_green_door"
                        elif self.map[i][j] == 'R':
                            item_type = "unlocked_red_door"
                        elif self.map[i][j] == 'Y':
                            item_type = "unlocked_yellow_door"
                        else:
                            # we dont have key :
                            cost[x][y] += -10000
                        neighbors = np.vstack(
                            (neighbors, [cost[x][y], item_type, item_index]))

                    elif self.map[i][j] == '1' or self.map[i][j] == '2' or self.map[i][j] == '3' or self.map[i][j] == '4':

                        # its GEM
                        if self.map[i][j] == '1':
                            item_type = "yellow_gem"
                            cost[x][y] += self.calc_gems_scores(
                                '1', self.agent.prev_gem)

                        elif self.map[i][j] == '2':
                            item_type = "green_gem"
                            cost[x][y] += self.calc_gems_scores(
                                '2', self.agent.prev_gem)

                        elif self.map[i][j] == '3':
                            item_type = "red_gem"
                            cost[x][y] += self.calc_gems_scores(
                                '3', self.agent.prev_gem)

                        elif self.map[i][j] == '4':
                            item_type = "blue_gem"
                            cost[x][y] += self.calc_gems_scores(
                                '4', self.agent.prev_gem)
                        cost[x][y] += self.find_path_for_gem_group(
                            best_gem_group, aim_index, np.array([x, y]))
                        item_index = (x, y)
                        neighbors = np.vstack(
                            (neighbors, [cost[x][y], item_type, item_index]))

                else:
                    cost[x][y] += -10000

                y += 1
            x += 1
        action = np.argmax(cost)
        max_score = np.amax(cost)
        for row in range(0, 9):
            if neighbors[i][0] == max_score:
                item_type = neighbors[i][1]
                item_index = neighbors[i][2]

        return action, item_type, item_index

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

    # Will be called only once.
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

    def find_path_for_gem_group(self, gem_group: np.array, gem_index: tuple, agent_index):
        # self.agent.agent_index[0] = (3, 0)

        find_path = FindPath(gem_group, gem_index, agent_index, self.agent.wall_indexes,
                             self.agent.barbed_indexes, self.agent.door_indexes, self.width, self.height)

        path = find_path.main()

        if type(path) != np.ndarray:
            if path == 0:
                cost = -10000

            elif path == -1:
                cost = -500

        else:
            cost = len(path) * -10
        return cost

    def remove_item_after_action(self, item_type: str, item_index: np.array):

        # delete this key from door indexes
        if item_type == "unlocked_green_door":
            if self.agent.green_key_number > 0:
                self.agent.green_key_number -= 1
        elif item_type == "unlocked_red_door":
            if self.agent.red_key_number > 0:
                self.agent.red_key_number -= 1
        elif item_type == "unlocked_yellow_door":
            if self.agent.yellow_key_number > 0:
                self.agent.yellow_key_number -= 1

        # delete this key from key indexes
        elif item_type == "get_green_key":
            self.agent.green_key_number += 1
        elif item_type == "get_red_key":
            self.agent.red_key_number += 1
        elif item_type == "get_yellow_key":
            self.agent.yellow_key_number += 1

        # delete this gem from gem indexes and set to the gem prev
        # delete from gem groups too
        elif item_type == "green_gem":
            self.agent.prev_gem = '1'
        elif item_type == "red_gem":
            self.agent.prev_gem = '2'
        elif item_type == "yellow_gem":
            self.agent.prev_gem = '3'
        elif item_type == "blue_gem":
            self.agent.prev_gem = '4'

        elif item_type == "barbed":
            pass
        elif item_type == "wall":
            pass
        elif item_type == "empty":
            pass

    def main(self):

        (action, item_type, item_index) = self.calc_neighbors(
            self.agent.agent_index[0][0], self.agent.agent_index[0][1])
        self.remove_item_after_action(item_type, item_index)

        if action == 0:
            return Action.UP_LEFT
        elif action == 1:
            return Action.UP
        elif action == 2:
            return Action.UP_RIGHT
        elif action == 3:
            return Action.LEFT
        elif action == 4:
            return Action.NOOP
        elif action == 5:
            return Action.RIGHT

        elif action == 6:
            return Action.DOWN_LEFT
        elif action == 7:
            return Action.DOWN

        elif action == 8:
            return Action.DOWN_RIGHT

        # return random.choice(
        #     [Action.DOWN, Action.DOWN_RIGHT, Action.DOWN_LEFT, Action.RIGHT, Action.LEFT, Action.UP_LEFT,
        #      Action.UP_RIGHT, Action.UP])
