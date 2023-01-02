import datetime
import numpy as np

from base import Action
from utils.config import GEMS


class ModelBasedPolicy:
    def __init__(self, agent):
        self.agent = agent
        self.height = self.agent.grid_height
        self.width = self.agent.grid_width
        self.map = np.array(self.agent.grid)
        self.threshold = 1e-3
        self.gamma = 0.9
        self.value_map = np.zeros((self.height, self.width), dtype=float)
        self.teleport = ['T', 'TA']
        self.barbed = ['*', '*A']
        self.slider = ['1', '2', '3', '4', 'g', 'r', 'y']
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT", "DOWN_RIGHT", "DOWN_LEFT", "UP_LEFT", "UP_RIGHT", "NOOP"]
        if 'list' not in self.agent.__dict__:
            self.agent.list = []
        if 'keys' not in self.agent.__dict__:
            self.agent.keys = {'r': 0,
                               'y': 0,
                               'g': 0}
        if 'prev_gem' not in self.agent.__dict__:
            self.agent.prev_gem = None

        self.agent.agent_index = self.get_agent_index()

        if 'prev_map' in self.agent.__dict__:
            self.calc_keys_count()
            self.calc_prev_gem()

        self.agent.prev_map = self.map

    @classmethod
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

    def calc_keys_count(self):
        keys = ["r", "g", "y"]
        x_agent, y_agent = self.agent.agent_index
        current_cell = self.agent.prev_map[x_agent][y_agent]
        if current_cell in keys:
            self.agent.keys[current_cell] += 1

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
        return [agent_index[0][0], agent_index[0][1]]

    def calc_reward(self, index) -> float:

        i_index, j_index = index
        reward = 0
        if self.map[i_index][j_index] == 'W':
            reward -= 500
        if self.map[i_index][j_index] == '1':
            reward += self.calc_gems_scores('1', self.agent.prev_gem)
        elif self.map[i_index][j_index] == '2':
            reward += self.calc_gems_scores('2', self.agent.prev_gem)
        elif self.map[i_index][j_index] == '3':
            reward += self.calc_gems_scores('3', self.agent.prev_gem)
        elif self.map[i_index][j_index] == '4':
            reward += self.calc_gems_scores('4', self.agent.prev_gem)
        elif self.map[i_index][j_index] == 'G':
            reward -= 500
            # todo
        elif self.map[i_index][j_index] == 'R':
            reward -= 500
            # todo
        elif self.map[i_index][j_index] == 'Y':
            reward -= 500
            # todo
        elif self.map[i_index][j_index] == 'g':
            reward += 10
        elif self.map[i_index][j_index] == 'r':
            reward += 10
        elif self.map[i_index][j_index] == 'y':
            reward += 10
        elif self.map[i_index][j_index] == '*':
            reward -= 20
        (i_agent, j_agent) = self.get_agent_index()
        if i_agent == i_index and j_agent != j_index:
            reward -= abs(j_agent - j_index)
        if j_agent == j_index and i_agent != i_index:
            reward -= abs(i_agent - i_index)
        if i_agent != i_index and j_agent != j_index:
            reward -= abs(i_agent - i_index) + -abs(j_agent - j_index)
        return reward

    def calc_probability(self, state, prob_action) -> float:
        (i_index, j_index) = state
        prob = 0
        for action in self.actions:
            x = i_index
            y = j_index
            if self.map[x][y] != 'W':
                if action == 'UP':
                    if i_index != 0:
                        x -= 1

                elif action == 'DOWN':
                    if i_index != self.height - 1:
                        x += 1

                elif action == 'LEFT':
                    if j_index != 0:
                        y -= 1

                elif action == 'RIGHT':
                    if j_index != self.width - 1:
                        y += 1

                elif action == 'DOWN_RIGHT':
                    if i_index != self.height - 1 and j_index != self.width - 1:
                        x += 1
                        y += 1

                elif action == 'DOWN_LEFT':
                    if i_index != self.height - 1 and j_index != 0:
                        x += 1
                        y -= 1

                elif action == 'UP_LEFT':
                    if i_index != 0 and j_index != 0:
                        x -= 1
                        y -= 1

                elif action == 'UP_RIGHT':
                    if i_index != 0 and j_index != self.width - 1:
                        x -= 1
                        y += 1

                prob += prob_action[action] * self.value_map[x][y]
            else:
                prob += 1 * self.value_map[x][y]

        return prob

    def is_normal(self, state) -> bool:
        return bool(state not in self.slider and state not in self.barbed and state not in self.teleport)

    def is_slider(self, state) -> bool:
        return state in self.slider

    def is_barbed(self, state) -> bool:
        return state in self.barbed

    def is_teleport(self, state) -> bool:
        return state in self.teleport

    def value_iteration(self):
        converge = False
        now1 = datetime.datetime.now()
        while not converge:
            delta = 0
            for i in range(0, self.height):
                for j in range(0, self.width):
                    temp = self.value_map[i][j]
                    total_probs = []
                    for action in self.actions:
                        state = (i, j)
                        total_prob = 0
                        if self.is_normal(self.map[i][j]):
                            total_prob = self.calc_probability(state, self.agent.probabilities['normal'][action])
                        elif self.is_slider(self.map[i][j]):
                            total_prob = self.calc_probability(state, self.agent.probabilities['slider'][action])
                        elif self.is_barbed(self.map[i][j]):
                            total_prob = self.calc_probability(state, self.agent.probabilities['barbed'][action])
                        elif self.is_teleport(self.map[i][j]):
                            total_prob = self.calc_probability(state, self.agent.probabilities['teleport'][action])

                        total_probs.append(total_prob)
                    self.value_map[i][j] = self.calc_reward((i, j)) + self.gamma * max(total_probs)
                    delta = max(delta, abs(temp - self.value_map[i][j]))
            # print("delta : ", delta)
            now2 = datetime.datetime.now()
            if (now2 - now1).total_seconds() > 0.9 or delta < self.threshold:
                converge = True
        print("value_map : ", self.value_map)
        # print("map : ", self.map)
        # if delta < self.threshold:
        #     converge = True

    def find_optimal_policy(self):
        (i_agent, j_agent) = self.get_agent_index()
        list = []
        count = 0
        for i in range(i_agent - 1, i_agent + 2):
            for j in range(j_agent - 1, j_agent + 2):
                if i != -1 and j != -1 and j != self.width and i != self.height:
                    list.append((count, self.value_map[i][j]))
                else:
                    list.append((count, -1000000))
                count += 1
        list.sort(key=lambda a: a[1], reverse=True)
        # print("policy : ", list) todo remove comment
        return list[0][0]

    def perform_action(self, action: int):
        agent_index = self.agent.agent_index
        x_agent, y_agent = agent_index
        if action == 0:
            if x_agent != 0 and y_agent != 0:
                return Action.UP_LEFT
        elif action == 1:
            if x_agent != 0:
                return Action.UP
        elif action == 2:
            if x_agent != 0 and y_agent != self.width - 1:
                return Action.UP_RIGHT
        elif action == 3:
            if y_agent != 0:
                return Action.LEFT
        elif action == 4:
            return Action.NOOP
        elif action == 5:
            if y_agent != self.width - 1:
                return Action.RIGHT
        elif action == 6:
            if x_agent != self.height - 1 and y_agent != 0:
                return Action.DOWN_LEFT
        elif action == 7:
            if x_agent != self.height - 1:
                return Action.DOWN
        elif action == 8:
            if x_agent != self.height - 1 and y_agent != self.width - 1:
                return Action.DOWN_RIGHT

        return Action.NOOP

    def calc_prev_gem(self):
        gems = ["1", "2", "3", "4"]
        x_agent, y_agent = self.agent.agent_index
        current_cell = self.agent.prev_map[x_agent][y_agent]
        if current_cell in gems:
            self.agent.prev_gem = current_cell

    def main(self):
        print("prev gem :", self.agent.prev_gem)
        print(self.map)
        # print(self.value_map)
        self.agent.list.append(self.agent.agent_gems[0])
        # print("prev gem :", self.agent.list)
        self.value_iteration()
        action = self.find_optimal_policy()
        return self.perform_action(action)
        # return random.choice(
        #     [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.DOWN_RIGHT, Action.DOWN_LEFT, Action.UP_LEFT,
        #      Action.UP_RIGHT, Action.NOOP])
