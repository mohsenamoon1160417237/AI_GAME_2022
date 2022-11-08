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
        self.wall = {'index': None, 'time': None}
        self.wall_stat = 0
        self.wall_types = []
        self.wall_directions = {'horiz': 0, 'vertic': 0}
        self.end_point_histories = []
        self.barbed = {'index': None, 'time': None}

    def search_obstacle_around_cells(self, wall: np.array) -> list:
        upper_cell, down_cell, left_cell, right_cell = [None, None, None, None]
        if wall[0] != 0:
            cell = [wall[0] - 1, wall[1]]
            if cell in self.wall_indexes.tolist():
                upper_cell = "wall"
            elif cell in self.barbed_indexes.tolist():
                upper_cell = "barbed"
            else:
                upper_cell = "empty"
        if wall[0] != self.height:
            cell = [wall[0] + 1, wall[1]]
            if cell in self.wall_indexes.tolist():
                down_cell = "wall"
            elif cell in self.barbed_indexes.tolist():
                down_cell = "barbed"
            else:
                down_cell = "empty"
        if wall[1] != 0:
            cell = [wall[0], wall[1] - 1]
            if cell in self.wall_indexes.tolist():
                left_cell = "wall"
            elif cell in self.barbed_indexes.tolist():
                left_cell = "barbed"
            else:
                left_cell = "empty"
        if wall[1] != self.width:
            cell = [wall[0], wall[1] + 1]
            if cell in self.wall_indexes.tolist():
                right_cell = "wall"
            elif cell in self.barbed_indexes.tolist():
                right_cell = "barbed"
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

    def get_latest_obstacle(self) -> Union[str, None]:
        newest_obstacle = None
        wall_index = self.wall['index']
        barbed_index = self.barbed['index']
        if wall_index is not None and barbed_index is None:
            newest_obstacle = "wall"
        elif wall_index is None and barbed_index is not None:
            newest_obstacle = "barbed"
        elif wall_index is not None and barbed_index is not None:
            obstacle_now = datetime.now()
            barbed_diff_time = obstacle_now - self.barbed['time']
            wall_diff_time = obstacle_now - self.wall['time']
            if barbed_diff_time < wall_diff_time:
                newest_obstacle = "barbed"
            else:
                newest_obstacle = "wall"

        return newest_obstacle

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

    def search_obstacles(self, path: dict, obstacle_index, obstacle_type: str) -> list:
        upper_cell, down_cell, left_cell, right_cell = self.search_obstacle_around_cells(obstacle_index)
        if obstacle_type == "wall":
            if right_cell == "wall" or left_cell == "wall":
                if self.wall_stat == 1:
                    self.wall_types.append("horiz")
                    self.wall_directions['horiz'] += 1
                elif self.wall_stat == 2:
                    if "vertic" in self.wall_types:
                        self.wall_types.append("horiz")
                        self.wall_directions['horiz'] += 1
                obstacle_row = obstacle_index[0]
                empty_cells = self.find_empty_cells_in_row(obstacle_row)
                nearest_cells = self.make_sorted_cells(empty_cells, path[-1, :], path)
                return nearest_cells
            if upper_cell == "wall" or down_cell == "wall":
                if self.wall_stat == 1:
                    self.wall_types.append("vertic")
                    self.wall_directions['vertic'] += 1
                elif self.wall_stat == 2:
                    if "horiz" in self.wall_types:
                        self.wall_types.append("vertic")
                        self.wall_directions['vertic'] += 1
                obstacle_col = obstacle_index[1]
                empty_cells = self.find_empty_cells_in_col(obstacle_col)
                nearest_cells = self.make_sorted_cells(empty_cells, path[-1, :], path)
                return nearest_cells
        elif obstacle_type == "barbed":
            if right_cell == "barbed" or left_cell == "barbed":
                obstacle_row = obstacle_index[0]
                empty_cells = self.find_empty_cells_in_row(obstacle_row)
                nearest_cells = self.make_sorted_cells(empty_cells, path[-1, :], path)
                return nearest_cells
            if upper_cell == "barbed" or down_cell == "barbed":
                obstacle_col = obstacle_index[1]
                empty_cells = self.find_empty_cells_in_col(obstacle_col)
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
            if path_index.tolist() in self.wall_indexes.tolist():
                self.wall_stat += 1
                self.wall['index'] = path_index
                self.wall['time'] = datetime.now()
                # for enum, history in enumerate(self.end_point_histories):
                #     agent_point, target_point = history[:2]
                #     if start_point == agent_point and end_point == target_point:
                #         self.end_point_histories[enum][2]['direct'] = 1
                return path
            elif path_index.tolist() in self.barbed_indexes.tolist():
                if tuple(path_index.tolist()) not in end_point:
                    self.barbed['index'] = path_index
                    self.barbed['time'] = datetime.now()
                    # self.barbed_histories.append(path_index)
                    return path

                path = np.vstack((path, path_index))
            else:
                path = np.vstack((path, path_index))
        self.tested_directions['diagonal'] = False

        latest_obstacle = self.get_latest_obstacle()
        if latest_obstacle == "wall":
            self.wall = {k: None for k in self.wall}
            self.wall_stat = 0
            self.wall_types = []
        else:
            self.barbed = {k: None for k in self.barbed}

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
            if path_index.tolist() in self.wall_indexes.tolist():
                # for enum, history in enumerate(self.end_point_histories):
                #     agent_point, target_point = history[:2]
                #     if start_point == agent_point and end_point == target_point:
                #         self.end_point_histories[enum][2]['diag'] = 1
                self.wall_stat += 1
                wall_index = self.wall.get('index')
                if wall_index is not None:
                    if path_index.tolist() == wall_index.tolist():
                        self.tested_directions['diagonal'] = True
                self.wall['index'] = path_index
                self.wall['time'] = datetime.now()
                return path
            elif path_index.tolist() in self.barbed_indexes.tolist():
                if tuple(path_index.tolist()) not in end_point:
                    barbed_index = self.barbed['index']
                    if barbed_index is not None:
                        self.tested_directions['diagonal'] = True
                    self.barbed['index'] = path_index
                    self.barbed['time'] = datetime.now()
                    return path

                path = np.vstack((path, path_index))
            else:
                path = np.vstack((path, path_index))

        latest_obstacle = self.get_latest_obstacle()
        if latest_obstacle == "wall":
            self.wall = {k: None for k in self.wall}
            self.wall_stat = 0
            self.wall_types = []
        else:
            self.barbed = {k: None for k in self.barbed}
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
                                                 end_point[-1])
            if self.wall['index'] is not None or self.barbed['index'] is not None or tuple(
                    path[-1, :].tolist()) == tuple(end_point[-1]):
                return path

            remaining_distance = abs(longer_distance - shorter_distance)
            path = self.find_ideal_direct_path(remaining_distance, distance, larger_col, larger_row, path, start_point,
                                               end_point[-1])
            return path
        elif tested_dir.get('diagonal') and not tested_dir.get('direct'):
            remaining_distance = abs(longer_distance - shorter_distance)
            path = self.find_ideal_direct_path(remaining_distance, distance, larger_col, larger_row, path, start_point,
                                               end_point[-1][0])
            return path

    def find_barbed_and_doors(self, path: np.array) -> dict:
        result = {'barbed_indexes': np.empty((0, 2)),
                  'doors': np.empty((0, 2)),
                  'path': path}
        for index in path:
            if index.tolist() in self.barbed_indexes.tolist():
                np.vstack((result['barbed_indexes'], index))
            if index.tolist() in self.door_indexes.tolist():
                np.vstack((result['barbed_indexes'], index))

        return result

    def main(self):
        now = datetime.now()
        path = np.array([self.agent_index])
        end_point = [[self.gem_index]]
        agent_index = self.agent_index.tolist()
        for wall_index in self.wall_indexes.tolist():
            if agent_index == wall_index:
                return 0
        while True:
            if (datetime.now() - now).total_seconds() > 0.5:
                return -1
            ideal_path = self.find_ideal_path(path, end_point)
            path = ideal_path
            if tuple(path[-1, :].tolist()) == self.gem_index:
                # print(self.end_point_histories)
                # return self.find_barbed_and_doors(path)
                return path

            latest_obstacle = self.get_latest_obstacle()
            if latest_obstacle == "wall":
                # if not self.tested_directions['diagonal']:
                targets = self.search_obstacles(ideal_path, self.wall['index'], "wall")
                if self.wall_directions['horiz'] > 2 or self.wall_directions['vertic'] > 2:
                    # print(self.end_point_histories)
                    return 0
                if targets is not None:
                    targets = [tuple(x.tolist()) for x in targets]
                    if len(targets) != 0:
                        if targets not in end_point:
                            end_point.append(targets)
            elif latest_obstacle == "barbed":
                targets = self.search_obstacles(ideal_path, self.barbed['index'], "barbed")
                if targets is not None:
                    targets = [tuple(x.tolist()) for x in targets]
                    if len(targets) != 0:
                        if targets not in end_point:
                            end_point.append(targets)
            # if len(end_point) == 0:
                # print("end point is empty")
            # print(f'end_point: {end_point}')
            if tuple(path[-1, :].tolist()) == tuple(end_point[-1][0]):
                end_point = end_point[:-1]
                if len(end_point) == 0:
                    end_point = self.gem_index
