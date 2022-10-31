import datetime

from base import BaseAgent, Action
from phase1 import Phase1


class Agent(BaseAgent):

    def do_turn(self) -> Action:
        now1 = datetime.datetime.now()
        phase1 = Phase1(self)
        print(self.gem_groups)
        action = phase1.main()
        now2 = datetime.datetime.now()
        print(f'total_seconds: {(now2 - now1).total_seconds()}')
        return action


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
