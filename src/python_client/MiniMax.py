import numpy as np
from base import  Action

class MiniMax:
    def __init__(self, agent, character):
        self.agent = agent
        self.map = np.array(self.agent.grid)
        self.height = self.agent.grid_height
        self.width = self.agent.grid_width
        if 'wall_indexes' not in self.agent.__dict__:
            self.agent.wall_indexes = self.make_wall_indexes()
        if 'gem_indexes' not in self.agent.__dict__:  # int
            self.agent.gem_indexes = self.make_gem_indexes()
        self.gem = ['1','2','3','4']
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT", "DOWN_RIGHT", "DOWN_LEFT", "UP_LEFT", "UP_RIGHT", "NOOP"]
        self.character = character
        self.visited_indexes = []
    def get_agent_index(self , character):
        agent_index = np.empty((0, 2), dtype=int)
        for row in range(self.map.shape[0]):
            if character =='A':
                agent = np.where(self.map[row] == 'EA')
                if len(agent[0]) != 0:
                    agent_index = np.vstack((agent_index, [row, agent[0][0]]))
            if character == 'B' :
                agent = np.where(self.map[row] == 'EB')
                if len(agent[0]) != 0:
                    agent_index = np.vstack((agent_index, [row, agent[0][0]]))
        return [agent_index[0][0], agent_index[0][1]]

    def make_gem_indexes(self) -> np.array:
        gem_indexes = np.empty((0, 3), dtype=int)  # row, col, gem_number
        for row in range(self.map.shape[0]):
            new_arr = np.where(self.map[row] == '1')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 1]))
            new_arr = np.where(self.map[row] == '2')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 2]))
            new_arr = np.where(self.map[row] == '3')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 3]))
            new_arr = np.where(self.map[row] == '4')
            for col in new_arr[0]:
                gem_indexes = np.vstack((gem_indexes, [row, col, 4]))

        return gem_indexes

    def make_wall_indexes(self) -> np.array:
        wall_indexes = []
        for row in range(self.height):
            for col in range(self.width):
                if self.map[row][col] == "W":
                    wall_indexes.append((row, col))
        return wall_indexes
    def is_terminal (self , state) -> bool:
        (i,j) = state
        if self.map[i][j] in self.gem :
            return True
        return False
    def transition_model(self , action , state) -> tuple:
        #return None for imposible action and wall
        # print(state)
        (i,j) = state
        next_state = ()
        if action == 'UP':
            if i != 0:
                next_state = (i-1,j)
            else:
                return None

        elif action == 'DOWN':
            if i != self.height - 1:
                next_state = (i+1,j)
            else:
                return None
        elif action == 'LEFT':
            if j != 0:
                next_state = (i,j-1)
            else:
                return None
        elif action == 'RIGHT':
            if j != self.width - 1:
                next_state = (i,j+1)
            else:
                return None
        elif action == 'DOWN_RIGHT':
            if i != self.height - 1 and j != self.width - 1:
                next_state = (i+1,j+1)
            else:
                return None
        elif action == 'DOWN_LEFT':
            if i != self.height - 1 and j != 0:
                next_state = (i+1 , j-1)
            else:
                return None
        elif action == 'UP_LEFT':
            if i != 0 and j != 0:
                next_state = (i-1,j-1)
            else:
                return None
        elif action == 'UP_RIGHT':
            if i != 0 and j != self.width - 1:
                next_state = (i-1,j+1)
            else:
                return None
        elif action == 'NOOP':
            next_state = (i,j)
        if next_state not in self.agent.wall_indexes :
            return next_state
        else:
            return None


    def minimax(self , action , state , max_turn , score) -> list:
        """"
        Main function
        """
        # print(action , state , max_turn , score)
        if self.is_terminal(state):
            score += 50
            return [action , state , max_turn , score]
        self.visited_indexes.append(state)
        if max_turn:
            list = []
            for action in self.actions :
                if self.transition_model(action , state) is not None and self.transition_model(action , state) not in self.visited_indexes:
                    state = self.transition_model(action , state)
                    print(action , state , max_turn , score)
                    list.append(self.minimax(action , state , False , score))
            list.sort(key=lambda a: a[3], reverse=True) #decrease
            print(list[0])
            return list[0]
        else:
            list = []
            for action in self.actions :
                if self.transition_model(action , state) is not None and self.transition_model(action , state) not in self.visited_indexes:
                    state = self.transition_model(action , state)
                    print(action , state , max_turn , score)
                    list.append(self.minimax(action , state , True , score))
            list.sort(key=lambda a: a[3], reverse=False) #increase
            # print(list[0])
            return list[0]
    def perform_action(self, action: str):
        if action == 'UP':
            return Action.UP

        elif action == 'DOWN':
            return Action.DOWN

        elif action == 'LEFT':
            return Action.LEFT

        elif action == 'RIGHT':
            return Action.RIGHT

        elif action == 'DOWN_RIGHT':
            return Action.DOWN_RIGHT

        elif action == 'DOWN_LEFT':
            return Action.DOWN_LEFT

        elif action == 'UP_LEFT':
            return Action.UP_LEFT

        elif action == 'UP_RIGHT':
            return Action.UP_RIGHT
        elif action == 'NOOP':
            return Action.NOOP

    # def heuristic(self, cur_index: tuple) -> int:
    #     """
    #     Calculates score of the current index. And if current index is equal to the self.goal_index(gem index) it should
    #     return the gem score.
    #     """
    #     return 10


    def main(self):
        action = 'NOOP'
        # print(self.agent_index)
        # print(state)
        # print(self.is_terminal((1,8)))
        if self.character == 'A':
            i = self.get_agent_index(self.character)[0]
            j = self.get_agent_index(self.character)[1]
            state = (i,j)
            max_turn = True
        else :
            i = self.get_agent_index(self.character)[0]
            j = self.get_agent_index(self.character)[1]
            state = (i,j)
            max_turn = False
        score = 0
        [action , state , max_turn , score] = self.minimax(action , state , max_turn , score)

        return self.perform_action(action)
