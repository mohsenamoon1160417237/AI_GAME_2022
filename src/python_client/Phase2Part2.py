import numpy as np
import gym

from Phase2 import Phase2


class Phase2Sarsa(Phase2):
    def __init__(self):
        super(Phase2, self).__init__()
        self.env = gym.make('FrozenLake-v0')
        self.epsilon = 0.9
        self.total_episodes = 10000
        self.max_steps = 100
        self.alpha = 0.85
        self.gamma = 0.95
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    # Function to choose the next action
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        return action

    # Function to learn the Q-value
    def update(self, state, state2, reward, action, action2):
        predict = self.Q[state, action]
        target = reward + self.gamma * self.Q[state2, action2]
        self.Q[state, action] = self.Q[state, action] + self.alpha * (target - predict)

    def main(self):
        # Starting the SARSA learning
        for episode in range(self.total_episodes):
            t = 0
            state1 = self.env.reset()
            action1 = self.choose_action(state1)

            while t < self.max_steps:
                # Visualizing the training
                self.env.render()

                # Getting the next state
                state2, reward, done, info = self.env.step(action1)

                # Choosing the next action
                action2 = self.choose_action(state2)

                # Learning the Q-value
                self.update(state1, state2, reward, action1, action2)

                state1 = state2
                action1 = action2

                # Updating the respective vaLues
                t += 1
                reward += 1

                # If at the end of learning process
                if done:
                    break
