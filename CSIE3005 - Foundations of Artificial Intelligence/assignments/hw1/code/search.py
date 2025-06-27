# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

from maze import Maze
import heapq


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def bfs(maze: Maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.getStart()
    objectives = maze.getObjectives()

    PARENT = {start: None}

    def get_path(node):
        path = []
        current = node
        while current:
            path.append(current)
            current = PARENT[current]

        path.reverse()
        return path

    fringe = [start]
    visited = set()

    while fringe:
        current = fringe.pop(0)

        neighbors = maze.getNeighbors(current[0], current[1])

        for neighbor in neighbors:
            if neighbor in objectives:
                return get_path(current) + [neighbor]

            if neighbor not in visited:
                fringe.append(neighbor)
                PARENT.setdefault(neighbor, current)
                visited.add(neighbor)

    # No path found
    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.getStart()
    objectives = maze.getObjectives()

    G_SCORES = {start: 0}

    def heuristic(start):
        # Manhattan distance
        goal = objectives[0]
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    F_SCORES = {start: heuristic(start)}

    PARENT = {start: None}

    def get_path(node):
        path = []
        current = node
        while current:
            path.append(current)
            current = PARENT[current]

        path.reverse()
        return path

    fringe = [start]

    while fringe:
        # pop the node with the smallest f
        fringe.sort(key=lambda x: F_SCORES[x])
        current = fringe.pop(0)

        neighbors = maze.getNeighbors(current[0], current[1])

        for neighbor in neighbors:
            if neighbor in objectives:
                return get_path(current) + [neighbor]

            tentaive_g = G_SCORES[current] + 1

            if G_SCORES.get(neighbor) == None or tentaive_g < G_SCORES[neighbor]:
                G_SCORES.setdefault(neighbor, tentaive_g)
                F_SCORES.setdefault(neighbor, tentaive_g + heuristic(neighbor))
                PARENT.setdefault(neighbor, current)

                if neighbor not in fringe:
                    fringe.append(neighbor)

    # No path found
    return []


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    def heuristic(start, remain_goals):
        # min Manhattan distance
        return (
            min(
                [
                    abs(start[0] - goal[0]) + abs(start[1] - goal[1])
                    for goal in remain_goals
                ]
            )
            if remain_goals
            else 0
        )

    class State:
        def __init__(self, pos, remain_goals, g_score, prev):
            self.pos = pos
            self.remain_goals = frozenset(remain_goals)
            self.g_score = g_score
            self.f_score = g_score + heuristic(pos, remain_goals)
            self.prev = prev

        def __eq__(self, other):
            return (
                self.position == other.position
                and self.remain_goals == other.remain_goals
            )

        def __str__(self):
            return f"{self.pos} remain_goals={len(self.remain_goals)}, g={self.g_score}, f={self.f_score}"

        def get_path(self):
            path = []
            current = self
            while current:
                path.append(current.pos)
                current = current.prev
            path.reverse()
            return path

    start = maze.getStart()
    all_goals = maze.getObjectives()

    start_state = State(start, all_goals, 0, None)
    STATE_SPACE = {(start_state.pos, start_state.remain_goals): start_state}

    FRINGE = []
    heapq.heappush(
        FRINGE, (start_state.f_score, (start_state.pos, start_state.remain_goals))
    )

    while FRINGE:
        _, key = heapq.heappop(FRINGE)
        current_state = STATE_SPACE[key]

        # goal test
        if len(current_state.remain_goals) == 0:
            return current_state.get_path()

        # expand
        tentaive_g = current_state.g_score + 1
        neighbors = maze.getNeighbors(current_state.pos[0], current_state.pos[1])
        for neighbor in neighbors:

            new_remain_goals = (
                current_state.remain_goals - {neighbor}
                if neighbor in current_state.remain_goals
                else current_state.remain_goals
            )
            # find neighbor state
            neighbor_state = STATE_SPACE.get((neighbor, new_remain_goals))

            if neighbor_state is None or tentaive_g < neighbor_state.g_score:
                new_state = State(neighbor, new_remain_goals, tentaive_g, current_state)

                STATE_SPACE.setdefault(
                    (new_state.pos, new_state.remain_goals), new_state
                )
                heapq.heappush(
                    FRINGE, (new_state.f_score, (new_state.pos, new_state.remain_goals))
                )

    return []


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.getStart()
    all_goals = maze.getObjectives()

    DISTANCE = {}

    def calculate_distance(pos):
        distance_to_goals = {}
        for goal in all_goals:
            distance_to_goals.setdefault(
                goal, abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            )
        DISTANCE.setdefault(pos, distance_to_goals)

    def heuristic(start, remain_goals):
        if len(remain_goals) == 0:
            return 0

        # calculate distance if not calculated before
        if DISTANCE.get(start) is None:
            calculate_distance(start)

        # 5.5s
        # h = sum([abs(start[0] - goal[0]) + abs(start[1] - goal[1]) for goal in remain_goals]) / len(remain_goals)

        # 4.2s
        # h = max([abs(start[0] - goal[0]) + abs(start[1] - goal[1]) for goal in remain_goals])

        # 4.0s
        max = 0
        for goal in remain_goals:
            if DISTANCE[start][goal] > max:
                max = DISTANCE[start][goal]
        h = max

        return h

    class State:
        def __init__(self, pos, remain_goals, g_score, prev):
            self.pos = pos
            self.remain_goals = frozenset(remain_goals)
            self.g_score = g_score
            self.f_score = g_score + heuristic(pos, remain_goals)
            self.prev = prev

        def __eq__(self, other):
            return (
                self.position == other.position
                and self.remain_goals == other.remain_goals
            )

        def __str__(self):
            return f"{self.pos} remain_goals={len(self.remain_goals)}, g={self.g_score}, f={self.f_score}"

        def get_path(self):
            path = []
            current = self
            while current:
                path.append(current.pos)
                current = current.prev
            path.reverse()
            return path

    start_state = State(start, all_goals, 0, None)
    STATE_SPACE = {(start_state.pos, start_state.remain_goals): start_state}

    FRINGE = []
    heapq.heappush(
        FRINGE, (start_state.f_score, (start_state.pos, start_state.remain_goals))
    )

    while FRINGE:
        _, key = heapq.heappop(FRINGE)
        current_state = STATE_SPACE[key]

        # goal test
        if len(current_state.remain_goals) == 0:
            return current_state.get_path()

        # expand
        tentaive_g = current_state.g_score + 1
        neighbors = maze.getNeighbors(current_state.pos[0], current_state.pos[1])
        for neighbor in neighbors:

            new_remain_goals = (
                current_state.remain_goals - {neighbor}
                if neighbor in current_state.remain_goals
                else current_state.remain_goals
            )
            # find neighbor state
            neighbor_state = STATE_SPACE.get((neighbor, new_remain_goals))

            if neighbor_state is None or tentaive_g < neighbor_state.g_score:
                new_state = State(neighbor, new_remain_goals, tentaive_g, current_state)
                STATE_SPACE.setdefault(
                    (new_state.pos, new_state.remain_goals), new_state
                )

                heapq.heappush(
                    FRINGE, (new_state.f_score, (new_state.pos, new_state.remain_goals))
                )

    return []


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.getStart()
    objectives = set(maze.getObjectives())

    visited = set()
    path = []

    def dfs(node):
        path.append(node)

        if node in objectives:
            visited.add(node)
        if visited == objectives:
            return -1

        childs = maze.getNeighbors(node[0], node[1])

        for child in childs:
            if child not in path:
                if dfs(child) == 1:
                    path.append(node)
                else:
                    return -1

        return 1

    dfs(start)

    return path
