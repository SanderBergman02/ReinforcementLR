import heapq
import math
import numpy as np
import time


def heuristic(point, goal):
    # Manhattan distance heuristic
    return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

def astar(grid, start, goal):
    open_set = []

    # Priority queue with (F-score, node)
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct the path and return
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            x, y = current[0] + dx, current[1] + dy
            neighbor = (x, y)

            # Assuming uniform cost for each step
            tentative_g = g_score[current] + 1

            if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0:
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current

    return None  # No path found

def pathfinding(size, blockers, head, fruit, game):
    matrix = np.zeros(size)
    # print(size)
    begin = (int(math.floor(head.y/20)), int(math.floor(head.x/20)))
    for i, item in enumerate(blockers):
        x = int(math.floor(item.x/20))
        y = int(math.floor(item.y/20))
        # print(blockers)
        if x != head.x and y != head.y:
            # print(item.x, item.y)
            matrix[y][x] = 1
        else:
            matrix[y][x] = 2

    end = (int(math.floor(fruit.y/20)), int(math.floor(fruit.x/20)))

    path = astar(matrix, begin, end)

    if begin[0] < 0 or begin[1] < 0:
        path = (())
    if game.is_collision(head) == True:
        path = (())

    if path == None:
        # print(blockers)
        # print(head)
        # print(end)
        matrix[begin[0]][begin[1]] = 2
        matrix[end[0]][end[1]] = 3
        # print(matrix)
        # print(path)
        # time.sleep(10)
    return path
