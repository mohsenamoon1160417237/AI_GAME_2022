from operator import ne
from base import Action
from utils.config import GEMS

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
        if 'wall_indexes' not in self.agent.__dict__:
            self.agent.wall_indexes = self.make_wall_indexes()
        if 'key_indexes' not in self.agent.__dict__:
            self.agent.key_indexes = self.make_key_indexes()
        if 'door_indexes' not in self.agent.__dict__:
            self.agent.door_indexes = self.make_door_indexes()
        if 'barbed_indexes' not in self.agent.__dict__:
            self.agent.barbed_indexes = self.make_barbed_indexes()
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

    def make_wall_indexes(self) -> np.array:
        agent_index = np.empty((0, 2), dtype=int)
        wall_indexes = np.empty((0, 2), dtype=int)  # row, col

        for row in range(self.map.shape[0]):
            agent = np.where(self.map[row] == 'EA')
            if len(agent[0]) != 0:
                agent_index = np.vstack((agent_index, [row, agent[0][0]]))
                self.agent.agent_index = agent_index
            new_arr = np.where(self.map[row] == 'W')
            for col in new_arr[0]:
                wall_indexes = np.vstack((wall_indexes, [row, col]))
        return wall_indexes
    def make_key_indexes(self) -> np.array:
        agent_index = np.empty((0, 2), dtype=int)
        key_indexes = np.empty((0, 3), dtype=int)  # row, col, key_color

        for row in range(self.map.shape[0]):
            agent = np.where(self.map[row] == 'EA')
            if len(agent[0]) != 0:
                agent_index = np.vstack((agent_index, [row, agent[0][0]]))
                self.agent.agent_index = agent_index
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

    def make_door_indexes(self) -> np.array:
        agent_index = np.empty((0, 2), dtype=int)
        door_indexes = np.empty((0, 3), dtype=int)  # row, col, door_color

        for row in range(self.map.shape[0]):
            agent = np.where(self.map[row] == 'EA')
            if len(agent[0]) != 0:
                agent_index = np.vstack((agent_index, [row, agent[0][0]]))
                self.agent.agent_index = agent_index
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

    def make_barbed_indexes(self) -> np.array:
        agent_index = np.empty((0, 2), dtype=int)
        barbed_indexes = np.empty((0, 2), dtype=int)  # row, col

        for row in range(self.map.shape[0]):
            agent = np.where(self.map[row] == 'EA')
            if len(agent[0]) != 0:
                agent_index = np.vstack((agent_index, [row, agent[0][0]]))
                self.agent.agent_index = agent_index
            new_arr = np.where(self.map[row] == '*')
            for col in new_arr[0]:
                barbed_indexes = np.vstack((barbed_indexes, [row, col]))
        return barbed_indexes
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
    def calc_neighbor(self , i_agent , j_agent) -> np.array :
        print("yess")
        neighbor = np.zeros((3,3), dtype=int)

        for i in range(i_agent-1 , i_agent+2):
            for j in range(j_agent-1,j_agent+2):
                if self.map[i][j] == 'E' :
                    if(i == i_agent) or (j == j_agent):
                        neighbor[i][j] += -1
                    else :
                        neighbor[i][j] += -2
                if self.map[i][j] == 'W' :
                    neighbor[i][j] += -10000
                if self.map[i][j] == 'g' or self.map[i][j] == 'r' or self.map[i][j] == 'y' :
                    neighbor[i][j] += 10
                if self.map[i][j] == '*' :
                    neighbor[i][j] += -20
                if self.map[i][j] == 'G' or self.map[i][j] == 'R' or self.map[i][j] == 'Y' :
                    # if we have key : 
                    neighbor[i][j] += 10
                    # we dont have key :
                    neighbor[i][j] += -10000    


                                                 



        return neighbor
        


    def main(self):
        self.calc_neighbor(self.agent.agent_index[0][0]  , self.agent.agent_index[0][1])
        
        return Action.NOOP
        # return random.choice(
        #     [Action.DOWN, Action.DOWN_RIGHT, Action.DOWN_LEFT, Action.RIGHT, Action.LEFT, Action.UP_LEFT,
        #      Action.UP_RIGHT, Action.UP])
