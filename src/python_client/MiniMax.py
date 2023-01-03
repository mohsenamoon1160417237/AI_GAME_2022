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

        self.agent.prev_map = self.map

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

    def calc_keys_count(self):
        keys = ["r", "g", "y"]
        x_agent = self.get_agent_index(self.character)[0]
        y_agent = self.get_agent_index(self.character)[0]
        current_cell = self.agent.prev_map[x_agent][y_agent]
        if current_cell in keys:
            self.agent.keys[current_cell] += 1

    def calc_prev_gem(self):
        gems = ["1", "2", "3", "4"]
        x_agent = self.get_agent_index(self.character)[0]
        y_agent = self.get_agent_index(self.character)[0]
        current_cell = self.agent.prev_map[x_agent][y_agent]
        if current_cell in gems:
            self.agent.prev_gem = current_cell

    def calc_gems_scores(cls, gem: str, prev_gem: str) -> int:
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

    def is_terminal(self, state_A, state_B) -> bool:
        # (i,j) = state
        # if self.map[i][j] in self.gem :
        #     return True
        # return False
        if state_A == (1, 2) or state_B == (1, 2):
            return True
        else:
            return False

    def minimax(self, action, state_A, state_B, max_turn, score) -> list:
        """"
        Main function
        """

        if self.is_terminal(state_A, state_B):
            print("terminate-------------------------------------------------")
            score += self.heuristic(state_A)
            print("ter", action, state_A, state_B, max_turn, score)
            return [action, state_A, state_B, max_turn, score]
        print("father", action, state_A, state_B, max_turn, score)
        if max_turn:
            self.visited_indexes_A.append(state_A)
            list = []
            init_state = state_A
            init_action = action
            for act in self.actions:
                if self.transition_model(act, init_state) is not None and self.transition_model(act, init_state) not in self.visited_indexes_A:
                    state_A = self.transition_model(act, init_state)
                    print("child max", init_state, act,
                          state_A, state_B, max_turn, score)
                    list.append(self.minimax(
                        act, state_A, state_B, False, score))
            if len(list) == 0:
                return [init_action, init_state, state_B, False, score]
            list.sort(key=lambda a: a[4], reverse=True)  # decrease
            return list[0]
        else:
            self.visited_indexes_B.append(state_B)
            list = []
            init_state = state_B
            init_action = action
            for act in self.actions:
                if self.transition_model(act, init_state) is not None and self.transition_model(act, init_state) not in self.visited_indexes_B:
                    state_B = self.transition_model(act, init_state)
                    list.append(self.minimax(
                        init_action, state_A, state_B, True, score))
            if len(list) == 0:
                return [init_action, state_A, init_state, False, score]
            list.sort(key=lambda a: a[4], reverse=False)  # increase
            return list[0]

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

    def heuristic(self, state_A) -> int:
        """
        Calculates score of the terminal state
        """
        if (state_A == (1, 2)):
            return +1
        else:
            return -1

    def main(self):
        action = 'NOOP'
        i = self.get_agent_index('A')[0]
        j = self.get_agent_index('A')[1]
        state_A = (i, j)
        i = self.get_agent_index('B')[0]
        j = self.get_agent_index('B')[1]
        state_B = (i, j)
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
        max_turn = True
        score = 0
        [action, state_A, state_B, max_turn, score] = self.minimax(
            action, state_A, state_B, max_turn, score)
        print("f", action, state_A, state_B, max_turn, score)
        return self.perform_action(action)
