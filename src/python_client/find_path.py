import numpy as np


class FindPath:
    def __init__(self, gem_group: np.array, gem_index: tuple, agent_index: np.array):
        self.gem_group = gem_group
        self.gem_index = gem_index
        # self.gem_arrange = gem_arrange
        self.agent_index = agent_index

    def find_ideal_path(self):
        path = np.empty((0, 2))
        agent_row, agent_col = self.agent_index
        gem_row, gem_col = self.gem_index
        row_diff = abs(agent_row - gem_row)
        col_diff = abs(agent_col - gem_col)
        if row_diff < col_diff:
            distance = "row"
            shorter_distance = row_diff
            longer_distance = col_diff
        else:
            distance = "col"
            shorter_distance = col_diff
            longer_distance = row_diff

        if agent_row > gem_row:
            larger_row = 'agent'
        else:
            larger_row = 'gem'
        if agent_col > gem_col:
            larger_col = 'agent'
        else:
            larger_col = 'gem'

        for x in range(shorter_distance):
            if larger_row == "gem":
                agent_row += 1
            else:
                agent_row -= 1
            if larger_col == "gem":
                agent_col += 1
            else:
                agent_col -= 1
            path = np.vstack((path, np.array([agent_row, agent_col])))

        remaining_distance = abs(longer_distance - shorter_distance)
        for y in range(remaining_distance):
            if distance == "row":
                if larger_col == "gem":
                    agent_col += 1
                else:
                    agent_col -= 1
            else:
                if distance == "col":
                    if larger_row == "gem":
                        agent_row += 1
                    else:
                        agent_row -= 1
            path = np.vstack((path, np.array([agent_row, agent_col])))

        print(path)
