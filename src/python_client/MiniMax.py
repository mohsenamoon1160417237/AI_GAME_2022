from base import Action
from utils.config import GEMS
import numpy as np


class MiniMax:
    def __init__(self, agent):
        self.agent = agent
        self.map = np.array(self.agent.grid)
        self.height = self.agent.grid_height
        self.width = self.agent.grid_width
        if 'wall_indexes' not in self.agent.__dict__:
            self.agent.wall_indexes = self.make_wall_indexes()
        if 'barbed_indexes' not in self.agent.__dict__:  # int
            self.agent.barbed_indexes = self.make_barbed_indexes()
        self.agent.gem_indexes = self.make_gem_indexes()
        self.gem = ['1', '2', '3', '4']
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT",
                        "DOWN_RIGHT", "DOWN_LEFT", "UP_LEFT", "UP_RIGHT", "NOOP"]
        self.character = self.agent.character
        self.visited_indexes_A = []
        self.visited_indexes_B = []
        if 'keys' not in self.agent.__dict__:
            self.agent.keys = {'r': 0,
                               'y': 0,
                               'g': 0}
        # if 'prev_gem' not in self.agent.__dict__:
        #     self.agent.prev_gem = None
        # if 'prev_gem_B' not in self.agent.__dict__:
        #     self.agent.prev_gem_B = None
        # if 'prev_map' in self.agent.__dict__:
        #     self.calc_keys_count()
        #     self.calc_prev_gem()


    def get_agent_index(self, character):
        agent_index = np.empty((0, 2), dtype=int)
        for row in range(self.map.shape[0]):
            if character == 'A':
                agent = np.where(self.map[row] == 'EA')
                if len(agent[0]) != 0:
                    agent_index = np.vstack((agent_index, [row, agent[0][0]]))
            if character == 'B':
                agent = np.where(self.map[row] == 'EB')
                if len(agent[0]) != 0:
                    agent_index = np.vstack((agent_index, [row, agent[0][0]]))
        return [agent_index[0][0], agent_index[0][1]]

    def make_gem_indexes(self) -> np.array:
        gem_indexes = np.empty((0, 2), dtype=int)  # row, col, gem_number
        for row in range(self.map.shape[0]):
            new_arr = np.where(self.map[row] == '1')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col]))
            new_arr = np.where(self.map[row] == '2')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col]))
            new_arr = np.where(self.map[row] == '3')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col]))
            new_arr = np.where(self.map[row] == '4')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col]))
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

    def make_wall_indexes(self) -> np.array:
        wall_indexes = []
        for row in range(self.height):
            for col in range(self.width):
                if self.map[row][col] == "W":
                    wall_indexes.append((row, col))
        return wall_indexes
    def make_barbed_indexes(self) -> np.array:
        barbed_indexes = np.empty((0, 2), dtype=int)  # row, col
        for row in range(self.map.shape[0]):
            new_arr = np.where(self.map[row] == '*')
            for col in new_arr[0]:
                barbed_indexes = np.vstack((barbed_indexes, [row, col]))
        return barbed_indexes
    def transition_model(self, action, state) -> tuple:
        # return None for imposible action and wall
        # print(state)
        (i, j) = state
        next_state = ()
        if action == 'UP':
            if i != 0:
                next_state = (i-1, j)
            else:
                return None

        elif action == 'DOWN':
            if i != self.height - 1:
                next_state = (i+1, j)
            else:
                return None
        elif action == 'LEFT':
            if j != 0:
                next_state = (i, j-1)
            else:
                return None
        elif action == 'RIGHT':
            if j != self.width - 1:
                next_state = (i, j+1)
            else:
                return None
        elif action == 'DOWN_RIGHT':
            if i != self.height - 1 and j != self.width - 1:
                next_state = (i+1, j+1)
            else:
                return None
        elif action == 'DOWN_LEFT':
            if i != self.height - 1 and j != 0:
                next_state = (i+1, j-1)
            else:
                return None
        elif action == 'UP_LEFT':
            if i != 0 and j != 0:
                next_state = (i-1, j-1)
            else:
                return None
        elif action == 'UP_RIGHT':
            if i != 0 and j != self.width - 1:
                next_state = (i-1, j+1)
            else:
                return None
        elif action == 'NOOP':
            next_state = (i, j)
        # i = self.get_agent_index('B')[0]
        # j = self.get_agent_index('B')[1]
        # if next_state == (i,j):
        #     return None
        if next_state not in self.agent.wall_indexes:
            return next_state
        else:
            return None

    def is_terminal(self) -> bool:
        self.agent.gem_indexes = self.make_gem_indexes()
        if len(self.agent.gem_indexes) ==0 :
            return True
        return False
        # if state_A == (1, 2) or state_B == (1, 2):
        #     return True
        # else:
        #     return False

    def find_best_action(self):
        best_score = -1000
        best_action = 'NOOP'
        i = self.get_agent_index('A')[0]
        j = self.get_agent_index('A')[1]
        state_A = (i, j)
        init_state = state_A
        i = self.get_agent_index('B')[0]
        j = self.get_agent_index('B')[1]
        state_B = (i, j)
        max_turn = False
        for action in self.actions :

            if self.transition_model(action , init_state) is not None :
                state_A = self.transition_model(action , init_state)
                self.visited_indexes_A = [init_state ]
                self.visited_indexes_B = []
                print('start :',init_state)
                print('action :' , action)
                score = self.minimax(state_A, state_B, max_turn)
                print('score : ',score)
                print(init_state , action, state_A, state_B, max_turn)
                print('------------------------------------------------------')
                if (score > best_score) :
                    best_action = action
                    best_score = score
        print("act : ",best_action)
        return best_action


    def minimax(self, state_A, state_B, max_turn) -> int:
        """"
        Main function
        """

        if self.is_terminal():
            # print("h :",self.heuristic())
            return self.heuristic()
        # print("father : " , state_A)
        if max_turn:
            best = -1000
            self.visited_indexes_A.append(state_A)
            init_state = state_A
            for act in self.actions:
                if self.transition_model(act, init_state) is not None and self.transition_model(act, init_state) not in self.visited_indexes_A:
                    state_A = self.transition_model(act, init_state)
                    # print("child : " , state_A)
                    # print("best before max:",best)
                    best =  max(best , self.minimax( state_A, state_B, not max_turn))
                    # print("best after max:",best)
            # print("best : " ,best)
            return best
        else:
            best = 1000
            self.visited_indexes_B.append(state_B)
            init_state = state_B
            for act in self.actions:
                if self.transition_model(act, init_state) is not None and self.transition_model(act, init_state) not in self.visited_indexes_B:
                    state_B = self.transition_model(act, init_state)
                    best = min(best , self.minimax( state_A, state_B, not max_turn))
            return best

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

    def heuristic(self) -> int:
        """
        Calculates score of the terminal state
        """
        return self.agent_A_score - self.agent_B_score
        # if (self.agent_A_score > self.agent_B_score):
        #     return +100
        # elif (self.agent_A_score < self.agent_B_score):
        #     return -100
        # else:
        #     0

    def main(self):
        action = 'NOOP'
        # i = self.get_agent_index('A')[0]
        # j = self.get_agent_index('A')[1]
        # state_A = (i, j)
        # i = self.get_agent_index('B')[0]
        # j = self.get_agent_index('B')[1]
        # state_B = (i, j)
        # print(self.agent_index)
        # print(state)
        # print(self.is_terminal())
        # init_state = state_A
        # for act in self.actions :
        #     print(act)
        #     print(init_state)
        #     if self.transition_model(act , init_state) is not None and self.transition_model(act , init_state) not in self.visited_indexes_A:
        #         state_A = self.transition_model(act , init_state)
        #         print(act , state_A)
        # max_turn = True
        # score = 0
        # [action, state_A, state_B, max_turn, score] = self.minimax(
        #     action, state_A, state_B, max_turn, score)
        # print("f", action, state_A, state_B, max_turn, score)
        action = self.find_best_action()
        # print(self.minimax((1,0), (1,4), False))
        return self.perform_action(action)
