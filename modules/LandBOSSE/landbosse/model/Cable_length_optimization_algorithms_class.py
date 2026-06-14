# All the algorithms related to minimum spanning tree have almost the same results
# Other approaches can have different results

from numpy import argmin, array, sqrt
import sys
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
from numpy import newaxis as na
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from landbosse.model.hybrid_heuristic import cable_design


class Cable_length_optimization_algorithms_class:

    Turbine_coordinates = None
    Substation_coordinate = None

# Input:
# The input to this python file is Turbine Coordinates and Substation coordinate.
# Example:
# Turbine_coordinates = np.array([(3.50,4.99),(3.70,3.20),(1.10,3.57),(3.37,3.58),(2.34,4.50),(2.53,2.01),(4.28,4.61),
#             (1.60,3.73),(3.50,4.23),(4.39,0.68),(0.16,4.35),(3.23,3.75),(0.54,0.57),(4.85,1.96),(0.99,2.34)])
# Substation_coordinate = np.array([(3.5,3.5)])
#
# Output
# Based on the coordinates given it gives the name to each coordinate based on its index number and then connect all the coordinates as shortly as possible to form a Minimum Spanning Tree.
# It shows the result as a dictionary.
# Example: Optimized Connection length : {(0, 6): 0.8676404785393547, (0, 8): 0.7599999999999998, (1, 3): 0.5032891812864647,
#                     (1, 13): 1.6911830178901393, (2, 7): 0.5249761899362676, (2, 10): 1.2214745187681977, (2, 14): 1.2349089035228469,
#                 (3, 11): 0.22022715545545243, (3, 15): 1.3868309197591464, (4, 7): 1.067941945987702, (4, 11): 1.163872845288522,
#                 (5, 15): 0.49091750834534326, (8, 11): 0.550726792520575, (9, 13): 1.360147050873544, (12, 14): 1.826307750626931}


    def __init__(self, Turbine_coordinates_input, Substation_coordinate_input, Turbine_Power):

        self.Turbine_coordinates  = Turbine_coordinates_input
        self.Substation_coordinate = Substation_coordinate_input
        self.Turbine_Power = Turbine_Power

    def Window_openMDAO(self):
        # all_coordinates = np.vstack((self.Turbine_coordinates, self.Substation_coordinate))
        # x_coordinates, y_coordinates = all_coordinates[:, 0], all_coordinates[:, 1]
        x = self.Turbine_coordinates[:,0]
        y = self.Turbine_coordinates[:,1]
        # for Sebastian's code
        cable_types = [[95, 300, 206], [120, 340, 221], [150, 375, 236], [185, 420, 256], [240, 480, 287], [300, 530, 316], [400, 590, 356], [500, 655, 406], [630, 715, 459], [800, 775, 521], [1000, 825, 579]]
        number_turbines_per_cable = [10,20,30]
        collection_voltage = 66000
        turbine_rated_current = np.divide(self.Turbine_Power*1e6, (collection_voltage  * np.sqrt(3.0)))
        layout = []
        for i in range(len(x)):
            layout.append([i,x[i],y[i]])
        # copied from "choose_cables" in hybrid_heuristic
        cables_info = cable_types
        cable_list = []
        for number in number_turbines_per_cable:
            for cable in cables_info:
                if turbine_rated_current * number <= cable[1]:
                    cable_list.append([number, cable[2]])
                    break
        # Optimization
        Opt_Cabling = cable_design(layout, self.Substation_coordinate, number_turbines_per_cable, cable_list)
        # transfer to fit Anidrudh style
        all_coordinates = np.vstack((self.Turbine_coordinates, self.Substation_coordinate))
        x_coordinates, y_coordinates = all_coordinates[:, 0], all_coordinates[:, 1]
        x = x_coordinates
        y= y_coordinates
        CablePlan = []
        Dist = []
        for v in Opt_Cabling[1][1]:
            for i in v:
                cur = [x-1 for x in i]
                if cur[0] < 0:
                    cur[0] = len(self.Turbine_coordinates)
                if cur[1] < 0:
                    cur[0] = len(self.Turbine_coordinates)
                CablePlan.append(tuple(cur))
                Dist.append(np.sqrt( (x[cur[0]] - x[cur[1]])**2 + (y[cur[0]] - y[cur[1]])**2 ))
        abc = {}
        for i in range(len(CablePlan)):
            abc[CablePlan[i]] = Dist[i]
        return abc
    
    def minimum_spanning_tree_Prim_algorithm(self):  # Prim's algorithm

        all_coordinates = np.vstack((self.Turbine_coordinates, self.Substation_coordinate))
        x_coordinates, y_coordinates = all_coordinates[:, 0], all_coordinates[:, 1]
        x = x_coordinates
        y= y_coordinates
        d_ij = np.hypot(x - x[:, na], y - y[:, na])
        Tcsr = minimum_spanning_tree(d_ij)
        sp = (Tcsr.toarray())
        return {tuple(i): sp[tuple(i)] for i in np.argwhere(sp)}


    def find_parent(self, parent, node):
        if parent[node] == node:
            return node
        return self.find_parent(parent, parent[node])

    def minimum_spanning_tree_kruskal_algorithm(self): # Kruskal algorithm
        all_coordinates = np.vstack((self.Turbine_coordinates, self.Substation_coordinate))
        num_nodes = all_coordinates.shape[0]
        edges = []

        # Create a list of all edges with their weights
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = np.linalg.norm(all_coordinates[i] - all_coordinates[j])
                edges.append((i, j, distance))

        # Sort edges by weight
        edges.sort(key=lambda x: x[2])

        parent = list(range(num_nodes))
        minimum_spanning_tree = {}

        for edge in edges:
            u, v, weight = edge
            parent_u = self.find_parent(parent, u)
            parent_v = self.find_parent(parent, v)

            if parent_u != parent_v:
                minimum_spanning_tree[(u, v)] = weight
                parent[parent_u] = parent_v

        return minimum_spanning_tree

    def calculate_cable_lengths_Dijkstra_Algorithm(self):  # Dijkstra's Algorithm
        all_coordinates = np.vstack((self.Turbine_coordinates, self.Substation_coordinate))
        num_nodes = all_coordinates.shape[0]
        distances = np.zeros((num_nodes, num_nodes))

        # Calculate distances between nodes (Euclidean distance)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = np.linalg.norm(all_coordinates[i] - all_coordinates[j])
                distances[i, j] = distances[j, i] = distance

        # Use Dijkstra's Algorithm to find the shortest paths
        shortest_paths = shortest_path(distances)

        # Calculate cable lengths for each connection
        cable_lengths = {}
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                cable_lengths[(i, j)] = shortest_paths[i, j]

        max_coordinate = max(max(key) for key in cable_lengths.keys())
        cable_lengths_ = {k: v for k, v in cable_lengths.items() if max_coordinate in k}

        return cable_lengths_

    def calculate_cable_lengths_TSP_Heuristic(self): # TSP Heuristic algorithm ( This is much more efficient when used in solving Saleman Problem)
        all_coordinates = np.vstack((self.Turbine_coordinates, self.Substation_coordinate))
        num_nodes = all_coordinates.shape[0]

        # Calculate pairwise distances between nodes (Euclidean distance)
        dist_matrix = distance_matrix(all_coordinates, all_coordinates)

        # Solve TSP using the linear_sum_assignment (Hungarian) algorithm
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # Calculate cable lengths for each connection based on the TSP solution
        cable_lengths = {}
        for i in range(num_nodes - 1):
            node1 = row_ind[i]
            node2 = row_ind[i + 1]
            cable_lengths[(node1, node2)] = dist_matrix[node1, node2]

        # Close the loop by connecting the last node to the starting node
        cable_lengths[(row_ind[-1], row_ind[0])] = dist_matrix[row_ind[-1], row_ind[0]]

        return cable_lengths


    def minimum_spanning_tree_boruvka_algorithm(self):  # Boruvka's Algorithm     (Don't use it take takes too much time)
        all_coordinates = np.vstack((self.Turbine_coordinates, self.Substation_coordinate))
        num_nodes = all_coordinates.shape[0]
        edges = []

        # Create a list of all edges with their weights
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = np.linalg.norm(all_coordinates[i] - all_coordinates[j])
                edges.append((i, j, distance))

        parent = list(range(num_nodes))
        minimum_spanning_tree = {}

        while len(parent) > 1:
            # Find the nearest neighbor for each component
            nearest_neighbors = {}
            for u, v, weight in edges:
                parent_u = self.find_parent(parent, u)
                parent_v = self.find_parent(parent, v)

                if parent_u != parent_v:
                    if u not in nearest_neighbors or weight < nearest_neighbors[u][1]:
                        nearest_neighbors[u] = (v, weight)
                    if v not in nearest_neighbors or weight < nearest_neighbors[v][1]:
                        nearest_neighbors[v] = (u, weight)

            # Add the minimum edge for each component to the MST
            for u, (v, weight) in nearest_neighbors.items():
                parent_u = self.find_parent(parent, u)
                parent_v = self.find_parent(parent, v)

                if parent_u != parent_v:
                    minimum_spanning_tree[(u, v)] = weight
                    parent[parent_u] = parent_v

        return minimum_spanning_tree

