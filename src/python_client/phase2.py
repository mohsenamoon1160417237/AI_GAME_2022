import datetime
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
        self.treshhold = 20
        self.gamma = 0.95
        self.teleport = ['T', 'TA']
        self.barbed = ['*', '*A']
        self.slider = ['1', '2', '3', '4', 'g', 'r', 'y']
        self.action = ["UP", "DOWN", "LEFT", "RIGHT",
                       "DOWN_RIGHT", "DOWN_LEFT", "UP_LEFT", "UP_RIGHT", "NOOP"]
        if 'list' not in self.agent.__dict__:
            self.agent.list = []
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
        # print(agent_index)
        return (agent_index[0][0], agent_index[0][1])

    def calc_reward(self, i_index, j_index) -> float:
        reward = 0
        if self.map[i_index][j_index] == 'W':
            reward += -500
        if self.map[i_index][j_index] == '1':
            reward += self.calc_gems_scores('1', self.agent.prev_gem)
        elif self.map[i_index][j_index] == '2':
            reward += self.calc_gems_scores('2', self.agent.prev_gem)
        elif self.map[i_index][j_index] == '3':
            reward += self.calc_gems_scores('3', self.agent.prev_gem)
        elif self.map[i_index][j_index] == '4':
            reward += self.calc_gems_scores('4', self.agent.prev_gem)
        elif self.map[i_index][j_index] == 'G':
            reward += -500
        #     # todo

        elif self.map[i_index][j_index] == 'R':
            reward += -500
        #     # todo

        elif self.map[i_index][j_index] == 'Y':
            reward += -500
        #     # todo
        elif self.map[i_index][j_index] == 'g':
            reward += 10
        elif self.map[i_index][j_index] == 'r':
            reward += 10
        elif self.map[i_index][j_index] == 'y':
            reward += 10
        elif self.map[i_index][j_index] == '*':
            reward += -20
        (i_agent, j_agent) = self.get_agent_index()
        if i_agent == i_index and j_agent != j_index:
            reward += - 1
        if j_agent == j_index and i_agent != i_index:
            reward += - 1
        if i_agent != i_index and j_agent != j_index:
            reward += - 2
        return reward

    def calc_probability(self, action, state, prob_action) -> float:
        (i_index, j_index) = state
        prob = 0
        for action in self.action:
            x = i_index
            y = j_index
            if action == 'UP':
                if i_index != 0:
                    x -= 1
                    prob += prob_action[action] * self.value_map[x][y]
            elif action == 'DOWN':
                if i_index != self.height - 1:
                    x += 1
                    prob += prob_action[action] * self.value_map[x][y]
            elif action == 'LEFT':
                if j_index != 0:
                    y -= 1
                    prob += prob_action[action] * self.value_map[x][y]
            elif action == 'RIGHT':
                if j_index != self.width - 1:
                    y += 1
                    prob += prob_action[action] * self.value_map[x][y]
            elif action == 'DOWN_RIGHT':
                if i_index != self.height - 1 and j_index != self.width - 1:
                    x += 1
                    y += 1
                    prob += prob_action[action] * self.value_map[x][y]
            elif action == 'DOWN_LEFT':
                if i_index != self.height - 1 and j_index != 0:
                    x += 1
                    y -= 1
                    prob += prob_action[action] * self.value_map[x][y]
            elif action == 'UP_LEFT':
                if i_index != 0 and j_index != 0:
                    x -= 1
                    y -= 1
                    prob += prob_action[action] * self.value_map[x][y]
            elif action == 'UP_RIGHT':
                if i_index != 0 and j_index != self.width - 1:
                    x -= 1
                    y += 1
                    prob += prob_action[action] * self.value_map[x][y]

            elif action == 'NOOP':
                prob += prob_action[action] * self.value_map[x][y]
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
        now1 = datetime.datetime.now()
        while not converge:
        # for k in range(5):
            delta = 0
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
            print("value_map : ", self.value_map)
            print("map : ", self.map)
            print("delta : ", delta)
            now2 = datetime.datetime.now()
            if (now2 - now1).total_seconds() > 0.95 or delta < self.treshhold :
                converge = True
            # if delta < self.treshhold:
            #     converge = True
    def find_optimal_policy(self):
        (i_agent, j_agent) = self.get_agent_index()
        list = []
        count = 0
        for i in range(i_agent - 1, i_agent + 2):
            for j in range(j_agent - 1, j_agent + 2):
                if i != -1 and j != -1 and j != self.width and i != self.height :
                    list.append((count , self.value_map[i][j]))
                else:
                    list.append((count , -1000000))
                count+=1
        list.sort(key=lambda a: a[1], reverse=True)
        print("policy : ",list)
        return list[0][0]


    def main(self):
        # print(self.map)
        # print(self.value_map)
        self.agent.list.append(self.agent.agent_gems[0])
        print("prev gem :", self.agent.list)
        self.value_iteration()
        action = self.find_optimal_policy()
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
        #     [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.DOWN_RIGHT, Action.DOWN_LEFT, Action.UP_LEFT,
        #      Action.UP_RIGHT, Action.NOOP])
