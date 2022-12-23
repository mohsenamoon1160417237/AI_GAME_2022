import datetime

from model_based_policy import ModelBasedPolicy
from reinforce_learning import ReinforceLearning
from base import BaseAgent, Action


class Agent(BaseAgent):
    # actions = [Action.RIGHT] + [Action.TELEPORT] * 100
    def do_turn(self) -> Action:
        now1 = datetime.datetime.now()
        phase2 = ModelBasedPolicy(self)
        # phase2 = ReinforceLearning(self)
        action = phase2.main()
        now2 = datetime.datetime.now()
        print(f'total_seconds: {(now2 - now1).total_seconds()}')
        return action


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
