import numpy as np
from typing import Union
from datetime import datetime


class FindPath:
    def __init__(self, gem_group: np.array, gem_index: tuple, agent_index: np.array, wall_indexes: np.array,
                 barbed_indexes: np.array, door_indexes: np.array, grid_width: int, grid_height: int):
        self.width = grid_width
        self.height = grid_height
        self.gem_group = gem_group
        self.gem_index = gem_index
        self.agent_index = agent_index
        self.agent_row, self.agent_col = self.agent_index
        # self.gem_arrange = gem_arrange
        self.wall_indexes = wall_indexes  # int
        self.barbed_indexes = barbed_indexes  # int
        self.door_indexes = door_indexes  # str
        self.tested_directions = {'direct': False, 'diagonal': False}
        self.wall = None
        self.wall_stat = 0
        self.wall_types = []
        self.wall_directions = {'horiz': 0, 'vertic': 0}
        self.end_point_histories = []

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
            if index not in self.wall_indexes.tolist():
                empty_cells = np.vstack((empty_cells, np.array(index)))

        best_cells = np.empty((0, 2))
        for cell in empty_cells:
            if cell.tolist() not in self.barbed_indexes.tolist():
                best_cells = np.vstack((best_cells, cell))
        if len(best_cells) == 0:
            return empty_cells
        else:
            return best_cells

    def find_empty_cells_in_col(self, col: int) -> np.array:
        empty_cells = np.empty((0, 2))
        for row in range(self.height):
            index = [row, col]
            if index not in self.wall_indexes.tolist():
                empty_cells = np.vstack((empty_cells, np.array(index)))

        best_cells = np.empty((0, 2))
        for cell in empty_cells:
            if cell.tolist() not in self.barbed_indexes.tolist():
                best_cells = np.vstack((best_cells, cell))
        if len(best_cells) == 0:
            return empty_cells
        else:
            return best_cells

    def make_sorted_cells(self, cells: np.array, start_point: np.array, path: np.array) -> Union[list, None]:
        distances = np.array([])
        row, col = start_point
        if len(cells) == 0:
            return None
        for cell in cells:
            dist_sum = np.sum([np.power((row - cell[0]), 2), np.power((col - cell[1]), 2)])
            dist = np.sqrt(dist_sum)
            distances = np.append(distances, dist)
        dist_dict = dict(zip(distances, cells))
        # dist_dict = {tuple(k): v for k, v in zip(distances.tolist(), cells.tolist())}
        sorted_dist_dict = dict(sorted(dist_dict.items(), key=lambda item: item[0]))
        sorted_cells = list(sorted_dist_dict.values())
        out_cells = []
        for cell in sorted_cells:
            if cell.tolist() not in path.tolist():
                out_cells.append(cell)
        return out_cells
        # cls.append(cells[index])
        # return cls

    def search_walls(self, path: dict) -> list:
        wall = self.wall
        upper_cell, down_cell, left_cell, right_cell = self.search_wall_around_cells(wall)
        if right_cell == "wall" or left_cell == "wall":
            if self.wall_stat == 1:
                self.wall_types.append("horiz")
                self.wall_directions['horiz'] += 1
            elif self.wall_stat == 2:
                if "vertic" in self.wall_types:
                    self.wall_types.append("horiz")
                    self.wall_directions['horiz'] += 1
            wall_row = wall[0]
            empty_cells = self.find_empty_cells_in_row(wall_row)
            nearest_cell = self.make_sorted_cells(empty_cells, path[-1, :], path)
            return nearest_cell
        if upper_cell == "wall" or down_cell == "wall":
            if self.wall_stat == 1:
                self.wall_types.append("vertic")
                self.wall_directions['vertic'] += 1
            elif self.wall_stat == 2:
                if "horiz" in self.wall_types:
                    self.wall_types.append("vertic")
                    self.wall_directions['vertic'] += 1
            wall_col = wall[1]
            empty_cells = self.find_empty_cells_in_col(wall_col)
            nearest_cells = self.make_sorted_cells(empty_cells, path[-1, :], path)
            return nearest_cells

    def find_ideal_direct_path(self, remaining_distance: int, distance: str, larger_col: str, larger_row: str,
                               path: np.array, start_point: tuple, end_point: tuple) -> np.array:
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
            if path_index.tolist() in self.wall_indexes.tolist() :
                self.wall_stat += 1
                self.wall = path_index
                for enum, history in enumerate(self.end_point_histories):
                    agent_point, target_point = history[:2]
                    if start_point == agent_point and end_point == target_point:
                        self.end_point_histories[enum][2]['direct'] = 1
                return path
            else:
                path = np.vstack((path, path_index))
        self.tested_directions['diagonal'] = False
        self.wall = None
        self.wall_stat = 0
        self.wall_types = []
        return path

    def find_ideal_diagonal_path(self, shorter_distance: int, larger_row: str, larger_col: str,
                                 path: np.array, start_point, end_point) -> np.array:
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
            if path_index.tolist() in self.wall_indexes.tolist()  :
                for enum, history in enumerate(self.end_point_histories):
                    agent_point, target_point = history[:2]
                    if start_point == agent_point and end_point == target_point:
                        self.end_point_histories[enum][2]['diag'] = 1
                self.wall_stat += 1
                if self.wall is not None:
                    if path_index.tolist() == self.wall.tolist():
                        self.tested_directions['diagonal'] = True
                self.wall = path_index
                return path
            else:
                path = np.vstack((path, path_index))

        self.wall_stat = 0
        self.wall = None
        self.wall_types = []
        return path

    def find_ideal_path(self, path: np.array, end_point: list) -> dict:
        tested_dir = self.tested_directions
        self.agent_row, self.agent_col = path[-1, :]
        start_point = (self.agent_row, self.agent_col)
        self.end_point_histories.append([start_point, end_point[-1][0], {'diag': 0, 'direct': 0}])
        end_row, end_col = end_point[-1][0]
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
            path = self.find_ideal_diagonal_path(shorter_distance, larger_row, larger_col, path, start_point,
                                                 end_point[-1][0])
            if self.wall is not None or tuple(path[-1, :].tolist()) == tuple(end_point[-1][0]):
                return path

            remaining_distance = abs(longer_distance - shorter_distance)
            path = self.find_ideal_direct_path(remaining_distance, distance, larger_col, larger_row, path, start_point,
                                               end_point[-1][0])
            return path
        elif tested_dir.get('diagonal') and not tested_dir.get('direct'):
            remaining_distance = abs(longer_distance - shorter_distance)
            path = self.find_ideal_direct_path(remaining_distance, distance, larger_col, larger_row, path, start_point,
                                               end_point[-1][0])
            return path

    def main(self):
        now = datetime.now()
        path = np.array([self.agent_index])
        end_point = [[self.gem_index]]
        agent_index = self.agent_index.tolist()
        for wall_indexes in self.wall_indexes.tolist():
            if agent_index == wall_indexes :
                return 0
        while True:
            if (datetime.now() - now).total_seconds() > 0.5:
                return -1
            ideal_path = self.find_ideal_path(path, end_point)
            path = ideal_path
            if tuple(path[-1, :].tolist()) == self.gem_index:
                # print(self.end_point_histories)
                return path
            if self.wall is not None:
                # if not self.tested_directions['diagonal']:
                targets = self.search_walls(ideal_path)
                if self.wall_directions['horiz'] > 9 or self.wall_directions['vertic'] > 9:
                    # print(self.end_point_histories)
                    return 0
                if targets is not None:
                    targets = [tuple(x.tolist()) for x in targets]
                    if targets not in end_point:
                        end_point.append(targets)
            if tuple(path[-1, :].tolist()) == tuple(end_point[-1][0]):
                end_point = end_point[:-1]
                if len(end_point) == 0:
                    end_point = self.gem_index
