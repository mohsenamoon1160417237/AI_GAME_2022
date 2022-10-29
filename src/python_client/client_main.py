from base import BaseAgent, Action
from phase1 import Phase1


class Agent(BaseAgent):

    def do_turn(self) -> Action:
        phase1 = Phase1(self)
        return phase1.main()


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
