import numpy as np


class FindPath:
    def __init__(self, gem_group: np.array, gem_index: tuple, agent_index: np.array, wall_indexes: np.array,
                 barbed_indexes: np.array, door_indexes: np.array, grid_width: int, grid_height: int):
        self.width = grid_width
        self.height = grid_height
        self.gem_group = gem_group
        self.gem_index = gem_index
        # self.gem_arrange = gem_arrange
        self.agent_index = agent_index
        self.wall_indexes = wall_indexes  # int
        self.barbed_indexes = barbed_indexes  # int
        self.door_indexes = door_indexes  # str
        self.tested_directions = {'direct': False, 'diagonal': False}
        self.wall = None
        self.agent_row, self.agent_col = [None, None]

    def search_wall_around_cells(self, wall: np.array) -> list:
        upper_cell, down_cell, left_cell, right_cell = [None, None, None, None]
        if wall[0] != 0:
            cell = [wall[0] - 1, wall[1]]
            if cell in self.wall_indexes.tolist():
                upper_cell = "wall"
            else:
                upper_cell = "empty"
        if wall[0] != self.height:
            cell = [wall[0] + 1, wall[1]]
            if cell in self.wall_indexes.tolist():
                down_cell = "wall"
            else:
                down_cell = "empty"
        if wall[1] != 0:
            cell = [wall[0], wall[1] - 1]
            if cell in self.wall_indexes.tolist():
                left_cell = "wall"
            else:
                left_cell = "empty"
        if wall[1] != self.width:
            cell = [wall[0], wall[1] + 1]
            if cell in self.wall_indexes.tolist():
                right_cell = "wall"
            else:
                right_cell = "empty"
        return [upper_cell, down_cell, left_cell, right_cell]

    def find_empty_cells_in_row(self, row: int) -> np.array:
        empty_cells = np.empty((0, 2))
        for col in range(self.width):
            index = [row, col]
            if index not in self.wall_indexes.tolist() and index not in self.barbed_indexes.tolist():
                empty_cells = np.vstack((empty_cells, np.array(index)))
        return empty_cells

    def find_nearest_cell(self, cells: np.array, start_point: np.array) -> np.array:
        distances = np.array([])
        row, col = start_point
        for cell in cells:
            dist_sum = np.sum([np.power((row - cell[0]), 2), np.power((col - cell[1]), 2)])
            dist = np.sqrt(dist_sum)
            distances = np.append(distances, dist)
        shortest_distance = np.min(distances)
        index = np.where(distances == shortest_distance)[0][0]
        return cells[index]

    def search_walls(self, path: dict) -> np.array:
        wall = self.wall
        upper_cell, down_cell, left_cell, right_cell = self.search_wall_around_cells(wall)
        if right_cell == "wall" or left_cell == "wall":
            wall_row = wall[0]
            empty_cells = self.find_empty_cells_in_row(wall_row)
            nearest_cell = self.find_nearest_cell(empty_cells, path[-1, :])
            return nearest_cell

    def find_ideal_direct_path(self, remaining_distance: int, distance: str, larger_col: str, larger_row: str,
                               path: np.array) -> np.array:
        for y in range(int(remaining_distance)):
            if distance == "row":
                if larger_col == "gem":
                    self.agent_col += 1
                else:
                    self.agent_col -= 1
            else:
                if distance == "col":
                    if larger_row == "gem":
                        self.agent_row += 1
                    else:
                        self.agent_row -= 1
            path_index = np.array([self.agent_row, self.agent_col])
            if path_index.tolist() in self.wall_indexes.tolist():
                self.wall = path_index
                return path
            else:
                path = np.vstack((path, path_index))
        self.tested_directions['diagonal'] = False
        self.wall = None
        return path

    def find_ideal_diagonal_path(self, shorter_distance: int, larger_row: str, larger_col: str,
                                 path: np.array) -> np.array:
        for x in range(int(shorter_distance)):
            if larger_row == "gem":
                self.agent_row += 1
            else:
                self.agent_row -= 1
            if larger_col == "gem":
                self.agent_col += 1
            else:
                self.agent_col -= 1
            path_index = np.array([self.agent_row, self.agent_col])
            if path_index.tolist() in self.wall_indexes.tolist():
                if self.wall is not None:
                    if path_index.tolist() == self.wall.tolist():
                        self.tested_directions['diagonal'] = True
                self.wall = path_index
                return path
            else:
                path = np.vstack((path, path_index))

        return path

    def find_ideal_path(self, path: np.array, end_point: np.array) -> dict:
        tested_dir = self.tested_directions
        self.agent_row, self.agent_col = path[-1, :]
        end_row, end_col = end_point
        row_diff = abs(self.agent_row - end_row)
        col_diff = abs(self.agent_col - end_col)
        if row_diff < col_diff:
            distance = "row"
            shorter_distance = row_diff
            longer_distance = col_diff
        else:
            distance = "col"
            shorter_distance = col_diff
            longer_distance = row_diff

        if self.agent_row > end_row:
            larger_row = 'agent'
        else:
            larger_row = 'gem'
        if self.agent_col > end_col:
            larger_col = 'agent'
        else:
            larger_col = 'gem'

        if not tested_dir.get('diagonal') and not tested_dir.get('direct'):
            path = self.find_ideal_diagonal_path(shorter_distance, larger_row, larger_col, path)
            if self.wall is not None or tuple(path[-1, :].tolist()) == end_point:
                return path

            remaining_distance = abs(longer_distance - shorter_distance)
            path = self.find_ideal_direct_path(remaining_distance, distance, larger_col, larger_row, path)
            return path
        elif tested_dir.get('diagonal') and not tested_dir.get('direct'):
            remaining_distance = abs(longer_distance - shorter_distance)
            path = self.find_ideal_direct_path(remaining_distance, distance, larger_col, larger_row, path)
            return path

    def main(self):
        path = np.array([self.agent_index])
        end_point = self.gem_index
        while True:
            ideal_path = self.find_ideal_path(path, end_point)
            path = ideal_path
            print(ideal_path)
            print(type(path[-1, :]))
            if tuple(path[-1, :].tolist()) == self.gem_index:
                return path
            if self.wall is not None:
                if not self.tested_directions['diagonal']:
                    end_point = tuple(self.search_walls(ideal_path))
            if tuple(path[-1, :].tolist()) == end_point:
                end_point = self.gem_index
