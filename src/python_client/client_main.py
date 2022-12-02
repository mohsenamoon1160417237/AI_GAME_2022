import datetime

from phase2 import Phase2
from base import BaseAgent, Action


class Agent(BaseAgent):
    # actions = [Action.RIGHT] + [Action.TELEPORT] * 100
    def do_turn(self) -> Action:
        now1 = datetime.datetime.now()
        phase2 = Phase2(self)
        action = phase2.main()
        now2 = datetime.datetime.now()
        print(f'total_seconds: {(now2 - now1).total_seconds()}')
        return action
        # return random.choice(
        #     [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.DOWN_RIGHT, Action.DOWN_LEFT, Action.UP_LEFT,
        #      Action.UP_RIGHT, Action.NOOP])


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
