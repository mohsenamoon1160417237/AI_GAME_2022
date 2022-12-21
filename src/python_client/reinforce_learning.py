import numpy as np
from typing import Union

from model_based_policy import ModelBasedPolicy
from base import Action

observation_space = ["empty", "wall", "barbed", "gem", "key", "locked"]
observation_space = np.array(observation_space)


class ReinforceLearning(ModelBasedPolicy):
    def __init__(self, agent):
        super().__init__(agent)
        self.epsilon = 0.9
        self.total_episodes = 10000
        self.max_steps = 100
        self.alpha = 0.85
        self.gamma = 0.95
        self.Q = np.zeros((self.height * self.width, len(self.actions)))
        self.reward = 0
        if 'action1' not in self.agent.__dict__:
            self.agent.action1 = None
        if 'action2' not in self.agent.__dict__:
            self.agent.action2 = None
        if 'state1' not in self.agent.__dict__:
            self.agent.state1 = self.agent.agent_index
        if 'state2' not in self.agent.__dict__:
            self.agent.state2 = self.agent.agent_index

    def make_state_num(self, state: tuple):
        i, j = state
        return i * self.width + j + 1

    @classmethod
    def calc_destination(cls, i, j, action):
        if action == 'UP':
            return i - 1, j
        elif action == 'UP_RIGHT':
            return i - 1, j + 1
        elif action == 'UP_LEFT':
            return i - 1, j - 1
        elif action == 'LEFT':
            return i, j - 1
        elif action == 'DOWN_LEFT':
            return i + 1, j - 1
        elif action == 'DOWN':
            return i + 1, j
        elif action == 'DOWN_RIGHT':
            return i + 1, j + 1
        elif action == 'RIGHT':
            return i, j + 1
        elif action == 'NOOP':
            return i, j

    # Function to choose the next action
    def choose_action(self, state: tuple) -> int:
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(range(9))
        else:
            action = np.argmax(self.Q[self.make_state_num(state), :])
        return action

    # Function to learn the Q-value
    def update(self, state, state2, reward, action, action2):
        state = self.make_state_num(state)
        state2 = self.make_state_num(state2)
        predict = self.Q[state, action]
        target = reward + self.gamma * self.Q[state2, action2]
        self.Q[state, action] = self.Q[state, action] + self.alpha * (target - predict)

    def calc_prob_reinforce(self, i, j, action1, action2) -> list:
        destination = self.calc_destination(i, j, action2)

        if self.is_normal(self.map[i][j]):
            prob = self.agent.probabilities['normal'][action1][action2]
            self_reward = prob * self.calc_reward(destination)
        elif self.is_slider(self.map[i][j]):
            prob = self.agent.probabilities['slider'][action1][action2]
            self_reward = prob * self.calc_reward(destination)
        elif self.is_barbed(self.map[i][j]):
            prob = self.agent.probabilities['barbed'][action1][action2]
            self_reward = prob * self.calc_reward(destination)
        else:  # is teleport
            prob = self.agent.probabilities['teleport'][action1][action2]
            self_reward = prob * self.calc_reward(destination)

        return [prob, self_reward]
        # i, j = destination
        # return {'index': (i, j), 'prob': prob, 'self': self_reward, 'children': []}

    def calc_forbidden_actions(self, i, j) -> list:
        forb_actions = []
        if i == 0:
            forb_actions.extend(["UP", "UP_LEFT", "UP_RIGHT"])
        if j == 0:
            forb_actions.extend(["LEFT", "DOWN_LEFT", "UP_LEFT"])
        if i == self.height - 1:
            forb_actions.extend(["DOWN", "DOWN_RIGHT", "DOWN_LEFT"])
        if j == self.width - 1:
            forb_actions.extend(["RIGHT", "DOWN_RIGHT", "UP_RIGHT"])

        return list(set(forb_actions))

    def calc_reward_reinforce(self, action) -> Union[float, int]:
        i, j = self.agent.agent_index
        forb_actions = self.calc_forbidden_actions(i, j)
        action = self.perform_action(action)
        tot_prob = 0
        tot_reward = 0

        if action == Action.UP:
            actions = [x for x in self.actions if x not in forb_actions]
            for act in actions:
                prob, reward = self.calc_prob_reinforce(i, j, "UP", act)
                tot_reward += reward
                tot_prob += prob
        elif action == Action.UP_LEFT:
            actions = [x for x in self.actions if x not in forb_actions]
            for act in actions:
                prob, reward = self.calc_prob_reinforce(i, j, "UP_LEFT", act)
                tot_reward += reward
                tot_prob += prob
        elif action == Action.UP_RIGHT:
            actions = [x for x in self.actions if x not in forb_actions]
            for act in actions:
                prob, reward = self.calc_prob_reinforce(i, j, "UP_RIGHT", act)
                tot_reward += reward
                tot_prob += prob
        elif action == Action.LEFT:
            actions = [x for x in self.actions if x not in forb_actions]
            for act in actions:
                prob, reward = self.calc_prob_reinforce(i, j, "LEFT", act)
                tot_reward += reward
                tot_prob += prob
        elif action == Action.DOWN_LEFT:
            actions = [x for x in self.actions if x not in forb_actions]
            for act in actions:
                prob, reward = self.calc_prob_reinforce(i, j, "DOWN_LEFT", act)
                tot_reward += reward
                tot_prob += prob
        elif action == Action.DOWN:
            actions = [x for x in self.actions if x not in forb_actions]
            for act in actions:
                prob, reward = self.calc_prob_reinforce(i, j, "DOWN", act)
                tot_reward += reward
                tot_prob += prob
        elif action == Action.DOWN_RIGHT:
            actions = [x for x in self.actions if x not in forb_actions]
            for act in actions:
                prob, reward = self.calc_prob_reinforce(i, j, "DOWN_RIGHT", act)
                tot_reward += reward
                tot_prob += prob
        elif action == Action.RIGHT:
            actions = [x for x in self.actions if x not in forb_actions]
            for act in actions:
                prob, reward = self.calc_prob_reinforce(i, j, "RIGHT", act)
                tot_reward += reward
                tot_prob += prob
        elif Action.NOOP:
            actions = [x for x in self.actions if x not in forb_actions]
            for act in actions:
                prob, reward = self.calc_prob_reinforce(i, j, "NOOP", act)
                tot_reward += reward
                tot_prob += prob

        if float(tot_prob) == float(0):
            return 0

        return tot_reward / tot_prob

    def calc_reward_reinforce_main(self, action):
        i, j = self.agent.agent_index
        action = self.perform_action(action)
        states = [{'index': (i, j),
                   'prob': 1,
                   'self': None,
                   'children': [],
                   'is_root': True,
                   'visited': False}
                  ]

        iter_states = iter(states)
        parent_states = states
        iter_parent_states = iter(parent_states)

        num = 1
        while True:
            try:
                state = next(iter_states)
            except StopIteration:
                if num == 5:
                    break
                try:
                    states = next(iter_parent_states)['children']
                except StopIteration:
                    states = parent_states[0]['children']
                    parent_states = states
                    iter_parent_states = iter(parent_states)
                    iter_states = iter(states)
                    num += 1
                continue

            state['children'] = self.calc_reward_reinforce(action, state)

        return states

    def calc_child_for_state(self, state, action):
        state['children'] = self.calc_reward_reinforce(action, state)
        return state

    def main(self):
        agent_index = self.agent.agent_index

        if self.agent.action1 is None:
            self.agent.action1 = self.choose_action(agent_index)
            self.reward = self.calc_reward_reinforce(self.agent.action1)
            return self.perform_action(self.agent.action1)

        if self.agent.action2 is not None:
            self.agent.state2 = agent_index
            self.update(self.agent.state1, self.agent.state2, self.reward, self.agent.action1, self.agent.action2)
            self.reward = self.calc_reward_reinforce(self.agent.action2)
            self.agent.action1 = self.agent.action2
            self.agent.state1 = self.agent.state2

        self.agent.action2 = self.choose_action(agent_index)
        return self.perform_action(self.agent.action2)

    def main2(self):
        agent_index = self.agent.agent_index
        self.agent.action1 = self.choose_action(agent_index)
        self.reward = self.calc_reward_reinforce_main(self.agent.action1)

    def test_main(self):
        agent_index = self.agent.agent_index
        if self.agent.action1 is None:
            # self.agent.action1 = self.choose_action(agent_index)
            self.reward = self.calc_reward_reinforce(1)
            return self.perform_action(1)
