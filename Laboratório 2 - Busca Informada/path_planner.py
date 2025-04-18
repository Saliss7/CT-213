from grid import Node, NodeGrid
from math import inf
import heapq


class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """
    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node
        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse
        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (successor_i, successor_j).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (successor_i, successor_j).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        self.node_grid.reset()

        start_node = self.node_grid.get_node(start_position[0], start_position[1])
        goal_node = self.node_grid.get_node(goal_position[0], goal_position[1])

        pq = []
        node = start_node
        node.g = 0
        node.f = 0
        heapq.heappush(pq, (node.f, node))
        while pq:
            f, node = heapq.heappop(pq)
            node_i, node_j = node.get_position()

            if node.closed:
                continue
            node.closed = True

            if (node_i, node_j) == goal_position:
                return self.construct_path(goal_node), node.g

            successor_positions = self.node_grid.get_successors(node_i, node_j)

            for successor_position in successor_positions:
                successor_i, successor_j = successor_position
                successor = self.node_grid.get_node(successor_i, successor_j)

                if successor.closed:
                    continue

                if successor.g > node.g + self.cost_map.get_edge_cost((node_i, node_j), (successor_i, successor_j)):
                    successor.g = node.g + self.cost_map.get_edge_cost((node_i, node_j), (successor_i, successor_j))
                    successor.f = successor.g  # in dijkstra f = g, there is no heuristic
                    successor.parent = node
                    heapq.heappush(pq, (successor.f, successor))
        return [], inf

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.

        :param start_position: position where the planning stars as a tuple (successor_i, successor_j).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (successor_i, successor_j).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        self.node_grid.reset()

        start_node = self.node_grid.get_node(start_position[0], start_position[1])
        goal_node = self.node_grid.get_node(goal_position[0], goal_position[1])

        pq = []
        node = start_node
        node.g = 0
        node.f = node.distance_to(goal_position[0], goal_position[1])
        heapq.heappush(pq, (node.f, node))
        while pq:
            f, node = heapq.heappop(pq)
            node_i, node_j = node.get_position()

            if node.closed:
                continue
            node.closed = True

            if (node_i, node_j) == goal_position:
                return self.construct_path(goal_node), node.g

            successor_positions = self.node_grid.get_successors(node_i, node_j)

            for successor_position in successor_positions:
                successor_i, successor_j = successor_position
                successor = self.node_grid.get_node(successor_i, successor_j)

                if successor.closed:
                    continue

                if successor.g > node.g + self.cost_map.get_edge_cost((node_i, node_j), (successor_i, successor_j)):
                    successor.g = node.g + self.cost_map.get_edge_cost((node_i, node_j), (successor_i, successor_j))
                    successor.f = successor.distance_to(goal_position[0], goal_position[1])
                    successor.parent = node
                    heapq.heappush(pq, (successor.f, successor))
        return [], inf

    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (successor_i, successor_j).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (successor_i, successor_j).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        self.node_grid.reset()

        start_node = self.node_grid.get_node(start_position[0], start_position[1])
        goal_node = self.node_grid.get_node(goal_position[0], goal_position[1])

        pq = []
        node = start_node
        node.g = 0
        node.f = node.distance_to(goal_position[0], goal_position[1])
        heapq.heappush(pq, (node.f, node))
        while pq:
            f, node = heapq.heappop(pq)
            node_i, node_j = node.get_position()

            if node.closed:
                continue
            node.closed = True

            if (node_i, node_j) == goal_position:
                return self.construct_path(goal_node), node.g

            successor_positions = self.node_grid.get_successors(node_i, node_j)

            for successor_position in successor_positions:
                successor_i, successor_j = successor_position
                successor = self.node_grid.get_node(successor_i, successor_j)

                if successor.closed:
                    continue

                if successor.g > node.g + self.cost_map.get_edge_cost((node_i, node_j), (successor_i, successor_j)):
                    successor.g = node.g + self.cost_map.get_edge_cost((node_i, node_j), (successor_i, successor_j))
                    successor.f = successor.g + successor.distance_to(goal_position[0], goal_position[1])
                    successor.parent = node
                    heapq.heappush(pq, (successor.f, successor))
        return [], inf
