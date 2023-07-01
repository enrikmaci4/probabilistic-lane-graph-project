import numpy as np
import matplotlib.pyplot as plt
import random
from math import inf
from inputs import *
import copy
import cmath
import functions.general as g
from classes.PLG import *


COLOUR_LOWER = 0
COLOUR_UPPER = 1
EMPTY_ENTRY = -1010101


class GraphPlotInformation:
    def __init__(self, PLG) -> None:
        # Node params
        self.node_size = NODE_SIZE
        self.node_colour = NODE_COLOUR
        self.colour_code_lanes_in_plg = COLOUR_CODE_LANES_IN_PLG
        # Edge params
        self.edge_line_width = EDGE_LINE_WIDTH
        self.edge_colour = EDGE_COLOUR
        # Edge shading params
        self.shade_edges_with_connection_probability = SHADE_EDGES_WITH_CONNECTION_PROBABILITY
        self.shade_darkness = 0.8
        # Node label params
        self.node_labels = NODE_LABELS
        self.node_labels_font_size = NODE_LABELS_FONT_SIZE
        self.node_labels_font_colour = NODE_LABELS_FONT_COLOUR
        # Conditonal params
        if self.colour_code_lanes_in_plg:
            self.node_colour = self.generate_colours_for_lane_ids(PLG)
        if self.node_labels is not None:
            self.node_labels = self.generate_node_labels(PLG)

    def generate_colours_for_lane_ids(self, PLG):
        lane_colour_dict = {}
        lane_colour_list = [0 for ii in range(PLG.num_nodes)]
        lane_ids = list(set(PLG.node_lane_ids))

        # Generate a colour (random tuple) of three value for each lane ID
        for lane_id in lane_ids:
            lane_colour_dict[lane_id] = (random.uniform(COLOUR_LOWER,COLOUR_UPPER), random.uniform(COLOUR_LOWER,COLOUR_UPPER), random.uniform(COLOUR_LOWER,COLOUR_UPPER))

        # Now create a list of colours corresponding to each lane ID
        for ii in range(PLG.num_nodes):
            lane_colour_list[ii] = lane_colour_dict[PLG.node_lane_ids[ii]]

        return lane_colour_list
    
    def generate_node_labels(self, PLG):
        # Get a vector of node labels
        if self.node_labels == "node_id":
            return [str(ii) for ii in range(PLG.num_nodes)]
        elif self.node_labels == "lane_id":
            return [str(PLG.node_lane_ids[ii]) for ii in range(PLG.num_nodes)]
        else:
            return self.node_labels


def draw(PLG):
    """Draws the PLG object.
    """
    # Initialise Graph Plot Information
    graph_plot_info = GraphPlotInformation(PLG)
    # Coordinates of nodes
    x = PLG.nodes[:,0]
    y = PLG.nodes[:,1]
    # Adjacency matrix
    adj_mat = PLG.adjmat

    # Get the shape of the adjacency matrix and assert that it is square
    shape_of_adj_mat = np.shape(adj_mat)
    num_rows = shape_of_adj_mat[0]
    num_cols = shape_of_adj_mat[1]
    assert num_rows == num_cols

    # Cycle through the adjacency matrix and plot edges
    for ii in range(num_rows):
        for jj in range(ii+1, num_cols):
            if max(adj_mat[ii,jj], adj_mat[jj,ii]) > 0:
                # If we've decided to shade the edges by probability then get
                # the shading for this edge
                if graph_plot_info.shade_edges_with_connection_probability:
                    shade_value = min(1 - adj_mat[ii, jj], 1 - adj_mat[jj, ii])*graph_plot_info.shade_darkness
                    graph_plot_info.edge_colour = [shade_value, shade_value, shade_value]

                # Plot the edge
                plt.plot([x[ii], x[jj]], [y[ii], y[jj]], color=graph_plot_info.edge_colour, linewidth=graph_plot_info.edge_line_width, zorder=3)

    # Plot the graph nodes
    plt.scatter(x, y, color=graph_plot_info.node_colour, s=graph_plot_info.node_size, zorder=4)

    # Plot the node labels
    if graph_plot_info.node_labels is not None:
        # Offset the labels slightly so that they are not plotted directly on
        # top of the nodes
        dx = 0.5
        dy = 0.5
        for ii in range(PLG.num_nodes):
            plt.text(x[ii] + dx, y[ii] + dy, graph_plot_info.node_labels[ii], color=graph_plot_info.node_labels_font_colour, fontsize=graph_plot_info.node_labels_font_size, fontweight="bold", zorder=5)


def arg_max_p_next_node(p_next_node, current_node, n_max=1):
    """Returns the next node with the highest probability of being visited
    given the current node"""
    if np.sum(p_next_node[current_node,:]) == 0:
        return None
    else:
        # Probability of transition
        p_transition = np.partition(p_next_node[current_node,:], -n_max)[-n_max]

        # If the probability is greater than 0 return True, otherwise return
        # False. So if we take the 2nd max node but it turns out there is no
        # 2nd max, we will return None.
        if p_transition > 0:
            return np.argpartition(p_next_node[current_node,:], -n_max)[-n_max]
        else:
            return None


def arg_max_p_next_node_given_target(p_next_node_given_target, p_next_node, closest_clusters_list, current_node, n_max=1):
    """Returns the next node with the highest probability of being visited
    given the current node and the target cluster. If we cannot find a next
    node given the target cluster then we will search for a next node given the
    next closest cluster, and so on.
    
    n_max - Get the nth max.
    """
    # Search for a node
    for target_cluster in closest_clusters_list:
        next_node = arg_max_p_next_node(p_next_node_given_target[target_cluster], current_node, n_max=n_max)
        if next_node:
            return next_node
        
    # If we get here it means we didn't find a node with the code above, try
    # p_next_node
    next_node = arg_max_p_next_node(p_next_node, current_node, n_max=n_max)
    

def path_generation(PLG: PLG, start_node: int, target_cluster: int, max_path_length=300):
    """Generates a path from the start node to the target cluster. If we reach
    a dead end then we will return a path that ends with "None"."""
    # Initialise the path
    path = [start_node]
    closest_clusters_list = PLG.closest_clusters_dict[target_cluster]
    
    # Continue to add nodes to the path until we reach the target cluster. If
    # We add "None" to the path then we have reached a dead end and should
    # stop. I.e. we have reached a node that has no outgoing edges.
    while (path[-1] not in PLG.target_clusters[target_cluster]) and \
          (len(path) < max_path_length) and \
          (path[-1] != None):
        # Get the next node
        next_node = arg_max_p_next_node_given_target(PLG.p_next_node_given_target, PLG.p_next_node, closest_clusters_list, path[-1])
        # Add the next node to the path
        path.append(next_node)

    # If the last node is None then remove it
    if path[-1] == None:
        path.pop(-1)

    # TODO: Could possibly remove repeated nodes so we don't need the
    # "max_path_length" parameter to break out of an infinite path loop
    # <code>

    return path


def path_tree_generation(PLG: PLG, target_cluster: int, path: list, paths: int, degree=2):
    """Generates a set of paths from the start node to the target cluster.
    we reach a dead end then we will return a path that ends with "None".
    
    target_cluster - Target cluster this vehicle is trying to get to.
    path           - A path from the starting node to the current node.
    paths          - A dictionary used to store the each paths in the tree of
                     possible paths this vehicle can take.
    degree         - At each node search 2 possible next nodes.
    """

    # Initialise the path
    closest_clusters_list = PLG.closest_clusters_dict[target_cluster]
    max_path_length = 10

    # Check if we can terminate this path, otherwise add a node
    if (path[-1] in PLG.target_clusters[target_cluster]) or \
       (len(path) >= max_path_length) or \
       (path[-1] == None):
        # Add the generated path to our set of possible list of paths
        paths[len(paths)] = path

    else:
        # Loop every neighbour and generate a path
        for ii in range(degree):
            # Get the next node
            next_node = arg_max_p_next_node_given_target(PLG.p_next_node_given_target, PLG.p_next_node, closest_clusters_list, path[-1], n_max=ii+1)
            # Recursively call into path_tree_generation and extend the path by
            # the next_node
            rc = path_tree_generation(PLG, target_cluster, path+[next_node], paths)

    return True


def fast_path_tree_generation(PLG: PLG, start_node: int, target_cluster: int, degree=2, max_path_length=20, max_lane_change=2, min_num_paths=3):
    """Generate a tree of possible paths.

    num_lanes_in_map
                   - The total number of lanes in the map.
    """
    # Initiailise some constants
    num_lanes_in_map = len(set(PLG.node_lane_ids))
    path = [start_node]
    paths = {}
    ii = 0

    # Call fast_path_generation_ until we generate enough paths, or until we
    # can't call it anymore.
    while (len(paths) < min_num_paths) and (max_lane_change <= num_lanes_in_map):
        # Initialisions. Yes - we initialise these two variables above aswell.
        # This is because we need the length in the while-statement so we have
        # to initialise them above. It's a bit ugly but that's ok :(
        path = [start_node]
        paths = {}
        
        # Generate paths
        rc = fast_path_tree_generation_(PLG, target_cluster, path, paths, max_lane_change=max_lane_change, degree=degree, max_path_length=max_path_length)
        max_lane_change += 1

    # We found some paths, now return them
    return paths


def fast_path_tree_generation_(PLG: PLG, target_cluster: int, path: list, paths: dict, degree=2, max_path_length=15, max_lane_change=2):
    """Generates a set of paths from the start node to the target cluster.
    we reach a dead end then we will return a path that ends with "None".

    In this version we add a max lane limitation to reduce the number of paths
    generated so we don't get silly paths that zig-zag between lanes.
    
    target_cluster - Target cluster this vehicle is trying to get to.
    path           - A path from the starting node to the current node.
    paths          - A dictionary used to store the each paths in the tree of
                     possible paths this vehicle can take.
    degree         - At each node search 2 possible next nodes.
    max_lane_change
                   - The maximum number of lane changes per path in our path
                     tree.
    min_num_paths  - Continue incrementing max_lane_change by 1 until we've
                     generated atleast this many paths.
    """
    # The first element of path, path[0], should always be an integer so check
    # that this is the case.
    assert type(path[0]) == int

    # Initialise some constants
    closest_clusters_list = PLG.closest_clusters_dict[target_cluster]
    last_element_is_none = False

    # Check the last element, it might be None
    if path[-1] == None:
        last_element_is_none = True
        path.pop(-1)

    # Now check the number of lane changes
    len_of_path = len(path)
    min_path_length_for_n_lane_changes = max_lane_change+1
    num_lane_changes = 0
    if len_of_path > min_path_length_for_n_lane_changes:
        # There has to be more than max_lane_change+1 nodes in the path for
        # the number of lane changes to exceed max_lane_change
        lane_ids = [PLG.node_lane_ids[path[ii]] for ii in range(len_of_path)]

        # Now cycle through the path and count the number of lane changes
        for ii in range(len_of_path-1):
            if lane_ids[ii] != lane_ids[ii+1]:
                num_lane_changes += 1

    # Check if we can terminate this path, otherwise add a node
    if (path[-1] in PLG.target_clusters[target_cluster]) or \
       (len(path) >= max_path_length) or \
       (last_element_is_none) or \
       (num_lane_changes > max_lane_change):
        
        # Check if we have any repeated nodes. This indicates the algorithm is
        # stuck in a loop generating the same two nodes over and over again.
        num_unique_nodes = len(set(path))

        # Only add the generated path to our set of possible list of paths if
        # it satisifies our lane changing constraints
        if (num_lane_changes <= max_lane_change) and (not last_element_is_none) and (num_unique_nodes == len_of_path):
            paths[len(paths)] = path

    else:
        # Loop every neighbour and generate a path
        for ii in range(degree):
            # Get the next node
            next_node = arg_max_p_next_node_given_target(PLG.p_next_node_given_target, PLG.p_next_node, closest_clusters_list, path[-1], n_max=ii+1)
            # Recursively call into path_tree_generation and extend the path by
            # the next_node
            rc = fast_path_tree_generation_(PLG, target_cluster, path+[next_node], paths, max_lane_change=max_lane_change, degree=degree, max_path_length=max_path_length)

    return True


def node_list_to_edge_phase(PLG: PLG, node_list: list):
    """Converts a list of nodes into a list of edge phases. The edge phases
    are the phases that are traversed when moving from one node to the next.
    If there are N nodes in the list then there will be N-1 edge phases.
    We will pad beggining of the phase list by repeating the first value once.
    """
    edge_phase_list = []
    num_nodes = len(node_list)

    # Assert that there are at least two nodes in the list
    assert num_nodes > 1

    # Cycle through the nodes and get the edge phase between each pair of 
    # nodes
    for ii in range(num_nodes-1):
        node_ii = node_list[ii]
        node_jj = node_list[ii+1]

        x_ii = PLG.nodes[node_ii, 0]
        y_ii = PLG.nodes[node_ii, 1]
        x_jj = PLG.nodes[node_jj, 0]
        y_jj = PLG.nodes[node_jj, 1]

        current_node = complex(x_ii, y_ii)
        next_node = complex(x_jj, y_jj)

        edge_phase = cmath.phase(next_node - current_node)
        edge_phase_list.append(edge_phase)

    # Pad the list by repeating the first element
    edge_phase_list.insert(0, edge_phase_list[0])

    return edge_phase_list


def node_path_to_output_data(PLG: PLG, node_path: list, mov_avg_win=3):
    """Converts a list of nodes into a 2D matrix of output data. The columns
    of the matrix are as follows:
    1. x coordinate of node
    2. y coordinate of node
    3. Heading angle of vehicle at node

    The heading angle is calculated by taking the phase of the vector that
    connects the current node to the next node. The heading angle is then
    smoothed using a moving average filter. The size of this moving average
    filter is given by mov_avg_win.
    """
    # Check if final node is None, if so then remove it
    if node_path[-1] == None:
        node_path = node_path[:-1]
        
    path_length = len(node_path)
    output_data = np.zeros((path_length, 3))
    edge_phase_list = node_list_to_edge_phase(PLG, node_path)

    # Assert that the path length is atleast mov_avg_win
    assert path_length >= mov_avg_win

    # Get the average heading angle for this path
    avg_edge_phase, _ = g.moving_average(edge_phase_list, n=mov_avg_win)

    # Check the length of the average edge phase list against the path length
    # and calculate the difference so we can pad the start of the
    # avg_edge_phase list with the first element
    avg_edge_phase_length = len(avg_edge_phase)
    diff = path_length - avg_edge_phase_length
    for ii in range(diff):
        avg_edge_phase = np.insert(avg_edge_phase, 0, avg_edge_phase[0])

    # Now stack the three columns of data into a single matrix
    output_data[:,0] = PLG.nodes[node_path, 0]
    output_data[:,1] = PLG.nodes[node_path, 1]
    output_data[:,2] = avg_edge_phase

    return output_data


def plot_node_path(PLG: PLG, node_path: list, color="red"):
    plt.plot(PLG.nodes[node_path, 0], PLG.nodes[node_path, 1], linewidth=2, color=color, zorder=10)
    return True

