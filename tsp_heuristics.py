#!/usr/bin/python3

import sys
import math
import time
import read_instance as ri

# install package 'progress' by running 'pip install progress'
from progress.bar import ShadyBar
from progress.spinner import Spinner

import plot_graph

############
# run this file as python .\tsp_heuristics.py .\instances\drei.txt 0 0
############


class tsp_class:

    # member variables (extend as needed)
    m_n = None  # number of nodes
    m_coords = None  # coordinates
    m_tour = None  # tour as list of vertices
    m_tour_improved = None  # tour as list of vertices improved by 2-opt
    m_val = None  # solution value
    m_val_improved = None  # solution value improved by 2-opt
    m_path = None  # tour as list of edges but all indices shifted -1

    def __init__(self, _n, _list):
        self.m_n = _n
        self.m_coords = _list
        # coords are handled as a list of tuples of coordinates (i.e. [(x1,y1), (x2,y2),...])
        self.m_coords = list(
            map(lambda x: (float(x[1]), float(x[2])), self.m_coords))
        self.m_distance_matrix = self.calculate_distance_matrix(
            self.get_coords())

        self.m_val = math.inf
        self.m_val_improved = self.m_val
        self.m_tour = []
        self.m_path = []

    def calculate_distance(self, vertex1, vertex2):
        """Calculates the euclidian distance between two vertices

        Takes a vertex as a tuple of coordinates '(x,y)'
        Returns a float
        """
        return float(math.sqrt((vertex1[0] - vertex2[0])**2 + (vertex1[1] - vertex2[1])**2))

    def calculate_distance_matrix(self, coords):
        """Calculates a 2D array of distances between all vertices"""
        distance_matrix = [[math.inf]*len(coords) for x in range(len(coords))]
        for i in range(len(coords)):
            for j in range(len(coords)):
                distance_matrix[i][j] = self.calculate_distance(
                    coords[i], coords[j])
        return distance_matrix

    def calculate_tour_distance(self, tour):
        """Calculates the value of a given tour

        Takes tour as list of vertices (i.e. [0,1,2,3,4,0])
        Returns: Value of the tour
        """
        distance_matrix = self.get_distance_matrix()
        sum_distance = sum([distance_matrix[tour[i]][tour[i+1]]
                            for i in range(len(tour) - 1)])
        return sum_distance

    def nearest_neighbour(self, coords, distance_matrix):
        """Find initial solution to TSP problem with Nearest Neighbour heuristik"""
        tour = []
        value = 0

        # add starting vertex to tour, remove from unvisited list
        unvisited_coords = [v for v in range(len(coords))]
        tour.append(unvisited_coords.pop(0))

        while unvisited_coords:
            min_value = sys.maxsize
            for i in unvisited_coords:
                min_value = min(min_value, distance_matrix[tour[-1]][i])
            # find vertex to be added next as unvisited vertex with minimum
            # distance to last vertex of current tour
            next_v = distance_matrix[tour[-1]].index(min_value)
            unvisited_coords.remove(next_v)
            tour.append(next_v)
        # close tour by returning to starting vertex
        tour.append(tour[0])
        self.set_tour(tour, improved=False)
        self.set_path(tour, construct_edges=True)
        self.m_val = self.calculate_tour_distance(tour)
        return self.m_val, self.m_tour

    def construct_adjacency_matrix(self, edges):
        """Constructs adjacency matrix of graph (helper function for Hierholzer Algorithm)
        Takes list of edges (i.e. [(0,1), (1,2), (2,0)])
        Returns adjacency matrix as list of list of adjacent vertices
        """
        adjacency_matrix = [[] for x in range(len(coords))]

        # for every edge, add second vertex to adjacency list of first vertex and vice versa
        for e in edges:
            adjacency_matrix[e[0][1]].append(e[0][0])
            adjacency_matrix[e[0][0]].append(e[0][1])
        return adjacency_matrix

    def construct_hamilton_circuit(self, eulerian_circuit):
        """Shortcut already visited vertices to get Hamilton circuit"""
        visited = set()
        visited_add = visited.add
        hamilton_circuit = [
            x for x in eulerian_circuit if x not in visited and not visited_add(x)]
        # last vertex should equal start vertex
        hamilton_circuit.append(hamilton_circuit[0])

        # construct hamilton circuit as list of edges
        hamiliton_edges = []
        for i in range(len(hamilton_circuit)-1):
            hamiliton_edges.append(
                (hamilton_circuit[i], hamilton_circuit[i+1]))
        return hamilton_circuit, hamiliton_edges

    def prim(self, coords, distance_matrix, start_node=0):
        """Algorithm of Prim to calculate minimum spanning tree

        Takes coords as a list of tuplse in the form of '(x,y)'
        Returns a list of edges of the MST in the form of tuples "(node1, node2)"
        """
        # initialize the priority queue as a list of dicts for each coordinate with
        # the vertex with its index and x- and y-coordinate
        # a dist value which saves the minimum distance connecting the vertex to the MST
        # and the parent vertex which is the closest vertex already in the MST
        # Format: [[index, distance, parent, (x-coord, y-coord)],...]
        priority_queue = [[v, sys.maxsize, math.inf, coords[v]]
                          for v in range(len(coords))]

        # initialize a MST list which holds the vertices in the order they are visited
        # initialize a edges list which holds the edges of the MST
        mst = []
        edges = []

        bar = ShadyBar('Constructing MST', max=len(coords))

        # initialize distance of first vertex as 0
        priority_queue[0][1] = 0
        # remove first vertex of priority queue and add to MST
        mst.append(priority_queue.pop(0))
        while priority_queue:
            for v in priority_queue:
                time_v = - time.process_time()
                for u in mst:
                    # if the current closest distance of v is bigger than the distance between u and v
                    if v[1] > distance_matrix[v[0]][u[0]]:
                        # update distance of v to the distance of u and v
                        v[1] = distance_matrix[v[0]][u[0]]
                        # and add u as the parent of v for the closest distance
                        v[2] = u[0]
                time_v += time.process_time()

            # get the vertex that has least dist in the priority queue
            new_vertex = min(priority_queue, key=lambda x: x[1])

            # removes new vertex from priority queue and adds to MST
            mst.append(priority_queue.pop(priority_queue.index(new_vertex)))

            # appends the edge connecting the new vertex and its parent vertex to the edges list
            edges.append([(new_vertex[0], new_vertex[2]),
                          distance_matrix[new_vertex[0]][new_vertex[2]]])

            bar.next()
        bar.finish()
        return mst, edges

    def hierholzer(self, adjacency_matrix):
        """Implements the Hierholzer Algorithm to find a eulerian circuit"""

        if not len(adjacency_matrix):
            return False

        # we start a sub-circuit with the first vertex
        sub_circuit = [0]
        eulerian_circuit = []

        while sub_circuit:
            # look at last vertex in sub circuit
            v = sub_circuit[-1]

            # if that vertex still has unvisited edges
            if adjacency_matrix[v]:
                # add unvisited edge to sub circuit
                next_vertex = adjacency_matrix[v].pop()
                sub_circuit.append(next_vertex)

            # backtrack to vertex in sub circuit that has unvisited edges
            else:
                # remove last vertex from sub circuit and add to eurlerian circuit
                eulerian_circuit.append(sub_circuit.pop())

        return eulerian_circuit

    def double_tree(self, coords):
        """Runs the double tree algorithm to solve the TSP"""
        # Calculate minimum spanning tree with Prim algorithm
        mst, edges = self.prim(
            coords=coords, distance_matrix=self.get_distance_matrix())

        # Calculate eulerian circuit with Hierholzer algorithm
        adjacency_matrix = self.construct_adjacency_matrix(edges)
        eulerian_circuit = self.hierholzer(adjacency_matrix)

        # Calculate hamilton circuit
        hamilton_circuit, hamiliton_edges = self.construct_hamilton_circuit(
            eulerian_circuit)

        self.set_tour(hamilton_circuit, improved=False)
        self.set_path(hamiliton_edges)
        self.m_val = self.calculate_tour_distance(hamilton_circuit)

        return hamilton_circuit, hamiliton_edges

    def optimize(self, method=1, improve=True):
        # method 0 -> use Nearest Neighbour
        if (method == 0):
            self.nearest_neighbour(coords=self.get_coords(
            ), distance_matrix=self.get_distance_matrix())
            self.draw_graph(ofile="NearestNeighbour_noImprove.png")
            if improve:
                self.opt2()
                self.draw_graph(ofile="NearestNeighbour_2opt.png")

        # method 1 -> use Double Tree
        elif (method == 1):
            self.double_tree(coords=self.get_coords())
            self.draw_graph(ofile="DoubleTree_noImprove.png")
            if improve:
                self.opt2()
                self.draw_graph(ofile="DoubleTree_2opt.png")

        else:
            print("unknown method")
            sys.exit()

    def opt2(self, improvement_threshold=0.05):
        """Run 2-opt algorithm to improve the initial solution

        improvement_threshold: specifies a threshold of improvement for the opt-2 algorithm to run again
        e.g. improvement_threshold=0.05 stop running 2-opt on the improved solution if the improvement
        of the solution value of the previous tour is less than 5%

        Replacing of two edges with two new edges is achieved by using the tour as a sequence of vertices
        and reversing a subsequence within the tour sequence (i.e. 0 1 2 3 4 5 0 -> 0 1 4 3 2 5 0: removes (1,2), (2,3))
        """
        # initial best value and tour (shift tour -1)
        tour = list(map(lambda x: x - 1, self.get_tour()))
        best_value = self.get_val()
        best_tour = tour
        improvement_factor = 1

        previous_best_value = best_value
        previous_best_tour = best_tour

        # animation of loading bar, length is only rough estimate
        bar = ShadyBar('Running 2-opt', max=len(tour)**2)

        # only run 2-opt if improvement is higher than specified improvement threshold
        while improvement_factor > improvement_threshold:
            for first_index in range(1, len(best_tour) - 2):
                for last_index in range(first_index + 1, len(best_tour) - 1):
                    # replace two existing edges with two new edges (more in docstring of this method)
                    new_tour = previous_best_tour[:first_index] + previous_best_tour[last_index -
                                                                                     1:first_index-1:-1] + previous_best_tour[last_index:]
                    new_value = self.calculate_tour_distance(new_tour)

                    # update the previous best value and tour if the new one is better
                    if previous_best_value > new_value:
                        previous_best_value = new_value
                        previous_best_tour = new_tour
                    bar.next()  # animation of loading bar
            improvement_factor = 1 - (previous_best_value/best_value)
            best_value = previous_best_value
            best_tour = previous_best_tour
            self.set_tour(best_tour)
            self.set_val(best_value, improved=True)
            self.set_path(best_tour, construct_edges=True)
        bar.finish()  # animation of loading bar
        return best_value, best_tour

    def draw_graph(self, ofile):
        """Creates plot of graph using the igraph package

        Takes file name of output file as string (i.e. "graph.png")
        """
        path = self.get_path()
        dist = [round(self.get_distance_matrix()[x[0]][x[1]], 2) for x in path]
        plot_graph.draw_g(self.get_n(), coords=self.get_coords(),
                          edges=path, path=path, dist=dist, ofile=ofile)

    def get_n(self):
        return self.m_n

    def get_coords(self):
        return self.m_coords

    def get_tour(self, improved=False):
        if improved:
            return self.m_tour_improved
        elif not improved:
            return self.m_tour

    def set_tour(self, tour, shift=True, improved=True):
        """Sets the tour but shifts every vertex + 1 bc our vertices start at 1 not 0"""
        tour = list(map(lambda x: x+1, tour))
        if improved:
            self.m_tour_improved = tour
        elif not improved:
            self.m_tour = tour

    def get_val(self, improved=False):
        if improved:
            return self.m_val_improved
        elif not improved:
            return self.m_val

    def set_val(self, value, improved=True):
        if improved:
            self.m_val_improved = value
        elif not improved:
            self.m_val = value

    def get_distance_matrix(self):
        return self.m_distance_matrix

    def get_path(self):
        return self.m_path

    def set_path(self, path, construct_edges=False):
        """Sets path as a list of edges
        If given path is a list of vertices (i.e. [0,1,2,3,0]), set construct_edges=True 
        """
        if construct_edges:
            path_edges = []
            value = 0
            for i in range(len(path)-1):
                path_edges.append((path[i], path[i+1]))
            self.m_path = path_edges
        else:
            self.m_path = path
# end class =======================================================


# main ============================================================
if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: tsp_apx.py file [m]")
        print("\t- file\t  input file ")
        print("\t- m\t  solution method\t (default 0: 0=heuristic1) ")
        sys.exit(-1)

    method = 0
    improve_opt = 0
    if (len(sys.argv) > 2):
        method = int(sys.argv[2])
        if (len(sys.argv) > 3):
            improve_opt = int(sys.argv[3])

    time_read = - time.process_time()
    n, coords = ri.readf(sys.argv[1])

    tsp_object = tsp_class(n, coords)
    time_read += time.process_time()

    time_solve = - time.process_time()
    if improve_opt == 0:
        tsp_object.optimize(method, improve=False)
    elif improve_opt == 1:
        tsp_object.optimize(method, improve=True)
    time_solve += time.process_time()

    print("time_read=", time_read)
    print("time_solve=", time_solve)
    print("val=", tsp_object.get_val())
    if improve_opt == 1:
        print("val improved=", tsp_object.get_val(improved=True))
    print("tour=\n", *tsp_object.get_tour())
    if improve_opt == 1:
        print("tour improved=\n", *tsp_object.get_tour(improved=True))


# end main ========================================================
