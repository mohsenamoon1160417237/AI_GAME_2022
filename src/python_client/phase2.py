import numpy as np
import random
from base import Action
from utils.config import GEMS


class Phase2:
    def __init__(self, Agent):
        self.agent = Agent
        self.height = self.agent.grid_height
        self.width = self.agent.grid_width
        self.map = np.array(self.agent.grid)
        self.treshhold = 1e-3
        self.gamma = 0.9
        self.teleport = ['T']
        self.barbed = ['*']
        self.slider = ['1', '2', '3', '4', 'g', 'r', 'y']
        self.action = ["UP", "DOWN", "LEFT", "RIGHT",
                       "DOWN_RIGHT", "DOWN_LEFT", "UP_LEFT", "UP_RIGHT", "NOOP"]
        if 'prev_gem' not in self.agent.__dict__:
            self.agent.prev_gem = None

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

    def calc_value(self, i_index, j_index):
        if self.map[i_index][j_index] == 'W':
            return -1000
        elif self.map[i_index][j_index] == '1':
            return self.calc_gems_scores('1', self.agent.prev_gem)
        elif self.map[i_index][j_index] == '2':
            return self.calc_gems_scores('2', self.agent.prev_gem)
        elif self.map[i_index][j_index] == '3':
            return self.calc_gems_scores('3', self.agent.prev_gem)
        elif self.map[i_index][j_index] == '4':
            return self.calc_gems_scores('4', self.agent.prev_gem)
        elif self.map[i_index][j_index] == 'G':
            # todo
            return -1000
        elif self.map[i_index][j_index] == 'R':
            # todo
            return -1000
        elif self.map[i_index][j_index] == 'Y':
            # todo
            return -1000
        elif self.map[i_index][j_index] == 'g':
            return 10
        elif self.map[i_index][j_index] == 'r':
            return 10
        elif self.map[i_index][j_index] == 'y':
            return 10
        elif self.map[i_index][j_index] == '*':
            return -20
        else:
            return 0

    def get_agent_index(self):
        agent_index = np.empty((0, 2), dtype=int)
        for row in range(self.map.shape[0]):
            agent = np.where(self.map[row] == 'EA')
            if len(agent[0]) != 0:
                agent_index = np.vstack((agent_index, [row, agent[0][0]]))
            agent = np.where(self.map[row] == 'TA')
            # it means agent in teleport
            if len(agent[0]) != 0:
                agent_index = np.vstack((agent_index, [row, agent[0][0]]))
            agent = np.where(self.map[row] == 'YA')
            if len(agent[0]) != 0:
                agent_index = np.vstack((agent_index, [row, agent[0][0]]))
            agent = np.where(self.map[row] == 'RA')
            if len(agent[0]) != 0:
                agent_index = np.vstack((agent_index, [row, agent[0][0]]))
            agent = np.where(self.map[row] == 'GA')
            if len(agent[0]) != 0:
                agent_index = np.vstack((agent_index, [row, agent[0][0]]))
            agent = np.where(self.map[row] == '*A')
            if len(agent[0]) != 0:
                agent_index = np.vstack((agent_index, [row, agent[0][0]]))
            # todo
        # print(agent_index)
        return (agent_index[0][0], agent_index[0][1])

    def calc_reward(self, i_index, j_index) -> float:
        (i_agent, j_agent) = self.get_agent_index()
        if i_agent == i_index and j_agent != j_index:
            return self.calc_value(i_index, j_index) - 1
        if j_agent == j_index and i_agent != i_index:
            return self.calc_value(i_index, j_index) - 1
        if i_agent != i_index and j_agent != j_index:
            return self.calc_value(i_index, j_index) - 2
        if i_agent == i_index and j_agent == j_index:
            return self.calc_value(i_index, j_index)

    def calc_probability(self, action, state, prob_action) -> float:
        (i_index, j_index) = state
        prob = 0
        for action in self.action:
            x = i_index
            y = j_index
            if action == 'UP':
                if i_index != 0:
                    x -= 1
                    prob += prob_action[action] + self.calc_value(x, y)
            elif action == 'DOWN':
                if i_index != self.height - 1:
                    x += 1
                    prob += prob_action[action] + self.calc_value(x, y)
            elif action == 'LEFT':
                if j_index != 0:
                    y -= 1
                    prob += prob_action[action] + self.calc_value(x, y)
            elif action == 'RIGHT':
                if j_index != self.width - 1:
                    y += 1
                    prob += prob_action[action] + self.calc_value(x, y)
            elif action == 'DOWN_RIGHT':
                if i_index != self.height - 1 and j_index != self.width - 1:
                    x += 1
                    y += 1
                    prob += prob_action[action] + self.calc_value(x, y)
            elif action == 'DOWN_LEFT':
                if i_index != self.height - 1 and j_index != 0:
                    x += 1
                    y -= 1
                    prob += prob_action[action] + self.calc_value(x, y)
            elif action == 'UP_LEFT':
                if i_index != 0 and j_index != 0:
                    x -= 1
                    y -= 1
                    prob += prob_action[action] + self.calc_value(x, y)
            elif action == 'UP_RIGHT':
                if i_index != 0 and j_index != self.width - 1:
                    x -= 1
                    y += 1
                    prob += prob_action[action] + self.calc_value(x, y)

            elif action == 'NOOP':
                prob += prob_action[action] + self.calc_value(x, y)
        return prob

    def isNormal(self, state) -> bool:
        if state not in self.slider:
            if state not in self.barbed:
                if state not in self.teleport:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def isSlider(self, state) -> bool:
        if state in self.slider:
            return True
        else:
            return False

    def isBarbed(self, state) -> bool:
        if state in self.barbed:
            return True
        else:
            return False

    def isTeleport(self, state) -> bool:
        if state in self.teleport:
            return True
        else:
            return False

    def value_iteration(self):
        converge = False
        self.value_map = np.zeros((self.height, self.width), dtype=float)
        while not converge:
            delta = 0
            # print("value_map : ",self.value_map)
            # print("map : ",self.map)
            for i in range(0, self.height):
                for j in range(0, self.width):
                    temp = self.value_map[i][j]
                    list = []
                    for action in self.action:
                        state = (i, j)
                        # print(action)
                        # print("state1 : ",state)
                        total_prob = 0
                        if self.isNormal(self.map[i][j]):
                            total_prob = self.calc_probability(
                                action, state, self.agent.probabilities['normal'][action])
                        elif self.isSlider(self.map[i][j]):
                            total_prob = self.calc_probability(
                                action, state, self.agent.probabilities['slider'][action])
                        elif self.isBarbed(self.map[i][j]):
                            total_prob = self.calc_probability(
                                action, state, self.agent.probabilities['barbed'][action])
                        elif self.isTeleport(self.map[i][j]):
                            total_prob = self.calc_probability(
                                action, state, self.agent.probabilities['teleport'][action])
                        # print(prob)
                        # print(self.calc_value(x,y))

                        list.append(total_prob)
                    self.value_map[i][j] = self.calc_reward(
                        i, j) + self.gamma * max(list)
                    delta = max(delta, abs(temp - self.value_map[i][j]))
            # print("delta : ",delta)
            if delta < self.treshhold:
                converge = True

    def main(self):
        # print(self.map)
        # print(self.value_map)
        self.value_iteration()
        return random.choice(
            [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.DOWN_RIGHT, Action.DOWN_LEFT, Action.UP_LEFT,
             Action.UP_RIGHT, Action.NOOP])
