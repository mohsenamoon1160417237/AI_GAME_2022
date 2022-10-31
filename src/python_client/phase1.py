import numpy as np

from base import Action
from utils.config import GEMS


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
        if 'gem_distances' not in self.agent.__dict__ and 'sorted_gem_distances' not in self.agent.__dict__:
            # float
            self.agent.sorted_gem_distances, self.agent.gem_distances = self.make_gem_distances()
        if 'gem_groups' not in self.agent.__dict__:  # int
            self.agent.gem_groups = self.group_gems()
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

    # Will be called only once.
    def make_gem_distances(self) -> list:
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
        return [sorted_distances, distances]

    def calc_neighbor(self, i_agent, j_agent) -> np.array:
        print("yess")
        neighbor = np.zeros((3, 3), dtype=int)

        for i in range(i_agent - 1, i_agent + 2):
            for j in range(j_agent - 1, j_agent + 2):
                if self.map[i][j] == 'E':
                    if (i == i_agent) or (j == j_agent):
                        neighbor[i][j] += -1
                    else:
                        neighbor[i][j] += -2
                if self.map[i][j] == 'W':
                    neighbor[i][j] += -10000
                if self.map[i][j] == 'g' or self.map[i][j] == 'r' or self.map[i][j] == 'y':
                    neighbor[i][j] += 10
                if self.map[i][j] == '*':
                    neighbor[i][j] += -20
                if self.map[i][j] == 'G' or self.map[i][j] == 'R' or self.map[i][j] == 'Y':
                    # if we have key : 
                    neighbor[i][j] += 10
                    # we dont have key :
                    neighbor[i][j] += -10000

        return neighbor

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
        # if len(gem_group) == len(searched_gems):
        #     gem_groups.append(gem_group)  # break the loop
        #     return gem_groups
        gem_group = self.search_nearby_cells(index, new_gem_indexes, gem_group)
        searched_gems = np.vstack((searched_gems, index))

        not_searched_gems = [x for x in gem_group.tolist() if x not in searched_gems.tolist()]
        if len(not_searched_gems) == 0:
            gem_groups.append(gem_group)
            return gem_groups
        index = not_searched_gems[0]
        gem_groups = self.search_gems(index, gem_group, searched_gems, gem_groups, gem_indexes)
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
                index_ls = [x for x in gem_indexes.tolist() if x not in [m for y in flatten_groups for m in y]]
                if len(index_ls) == 0:
                    return gem_groups
                index = index_ls[0]
            gem_group = np.vstack((gem_group, index))
            gem_groups = self.search_gems(index, gem_group, searched_gems, gem_groups, gem_indexes)

    def main(self):
        # self.calc_neighbor(self.agent.agent_index[0][0], self.agent.agent_index[0][1])
        return Action.NOOP
        # return random.choice(
        #     [Action.DOWN, Action.DOWN_RIGHT, Action.DOWN_LEFT, Action.RIGHT, Action.LEFT, Action.UP_LEFT,
        #      Action.UP_RIGHT, Action.UP])
