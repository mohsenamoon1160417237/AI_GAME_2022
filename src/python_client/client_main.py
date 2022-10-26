import random
from base import BaseAgent, Action
from do_turn import Phase1


class Agent(BaseAgent):

    def do_turn(self) -> Action:
        # phase1 = Phase1(self.grid_width, self.grid_width)
        # return phase1.main()
        return Action.NOOP


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
