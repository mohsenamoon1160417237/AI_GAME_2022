import numpy as np
import datetime

now1 = datetime.datetime.now()
arr = np.array([['EA', 'E', 'E', 'E', 'E', 'E', 'E', 'E', '*', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E'],
                ['E', '1', '2', 'y', 'G', 'E', 'E', 'E', 'E', 'E', '2', 'r', 'E', 'E', 'E', 'R', 'E', 'E', 'E', 'E'],
                ['E', 'E', '3', 'E', 'E', 'E', 'G', 'G', 'E', 'E', '1', 'E', 'g', 'E', 'E', 'G', 'R', 'Y', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'r', 'E', 'E', 'E', 'E', 'R', 'R', 'E', 'E', 'E', 'E', 'E', 'E', '*', 'E', 'E'],
                ['E', 'E', 'E', 'E', '3', 'E', 'E', 'Y', 'Y', 'E', 'E', '2', 'E', 'y', 'E', 'E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', '*', 'E', 'E', '2', 'E', '1', '1', '1', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E'],
                ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'g', 'r', 'y', 'E', 'E', 'E', 'E', 'E', 'E']])

agent_index = np.empty((0, 2), dtype=int)

gem_indexes = np.empty((0, 3), dtype=int)  # row, col, gem_number
for row in range(arr.shape[0]):
    agent = np.where(arr[row] == 'EA')
    if len(agent[0]) != 0:
        agent_index = np.vstack((agent_index, [row, agent[0][0]]))
    new_arr = np.where(arr[row] == '1')
    for col in new_arr[0]:
        gem_indexes = np.vstack((gem_indexes, [row, col, 1]))
    new_arr = np.where(arr[row] == '2')
    for col in new_arr[0]:
        gem_indexes = np.vstack((gem_indexes, [row, col, 2]))
    new_arr = np.where(arr[row] == '3')
    for col in new_arr[0]:
        gem_indexes = np.vstack((gem_indexes, [row, col, 3]))
    new_arr = np.where(arr[row] == '4')
    for col in new_arr[0]:
        gem_indexes = np.vstack((gem_indexes, [row, col, 4]))
print(gem_indexes)

distances = np.array([])

agent_row = agent_index[0][0]
agent_col = agent_index[0][1]

for index in gem_indexes:
    row, col = index[0:2]

# for index in gem_indexes:
#     row = index[0]
#     col = index[1]
#     dist_sum = np.sum([np.power((row - agent_row), 2), np.power((col - agent_col), 2)])
#     distance = np.sqrt(dist_sum)
#     distances = np.append(distances, distance)
# print(distances)
# sorted_distances = np.sort(distances)
# print(f'min distance is {sorted_distances[0]}')
now2 = datetime.datetime.now()
print(f'total seconds: {(now2 - now1).total_seconds()}')
