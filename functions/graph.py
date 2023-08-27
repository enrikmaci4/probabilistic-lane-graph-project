import numpy as np
import matplotlib.pyplot as plt
import random
import math
from inputs import *
from fnames import *
import copy
import cmath
import functions.general as g
from classes.PLG import *


COLOUR_LOWER = 0
COLOUR_UPPER = 1
EMPTY_ENTRY = -1010101
HEAD_ANG_MOV_AVG_WIN = 3
# NOTE: This needs to be a high positive number
INF = 1000
INF_SMALL = 1E-10


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

    # Plot artificial edges
    if PLOT_ARTIFICIAL_CONNECTIONS:
        _draw(connect_nodes(PLG), PLG, color="red")


def _draw(adj_mat, PLG_: PLG, color: str):
    """Plot graph specified in colour specified
    """
    # Initialise Graph Plot Information
    graph_plot_info = GraphPlotInformation(PLG_)
    # Coordinates of nodes
    x = PLG_.nodes[:,0]
    y = PLG_.nodes[:,1]

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
                plt.plot([x[ii], x[jj]], [y[ii], y[jj]], color=color, linewidth=graph_plot_info.edge_line_width, zorder=3)



def connect_nodes(PLG_: PLG):
    """Artificially connect nodes
    """
    # Initialie another matrix
    PLG_adverse_conn = np.zeros((PLG_.num_nodes, PLG_.num_nodes))

    # Create connections
    PLG_adverse_conn[1200,173] = 1
    PLG_adverse_conn[1201,171] = 1
    PLG_adverse_conn[1202,170] = 1
    PLG_adverse_conn[1203,169] = 1
    PLG_adverse_conn[1419,168] = 1
    PLG_adverse_conn[1204,167] = 1
    PLG_adverse_conn[1205,166] = 1
    PLG_adverse_conn[1206,165] = 1
    PLG_adverse_conn[1207,164] = 1
    PLG_adverse_conn[1208,162] = 1
    PLG_adverse_conn[1210,160] = 1
    PLG_adverse_conn[1212,158] = 1
    PLG_adverse_conn[1214,524] = 1
    PLG_adverse_conn[1256,523] = 1

    PLG_adverse_conn[391,736] = 1
    PLG_adverse_conn[1312,735] = 1
    PLG_adverse_conn[390,736] = 1

    PLG_adverse_conn[32,1356] = 1
    PLG_adverse_conn[30,1051] = 1

    PLG_adverse_conn[403,485] = 1
    PLG_adverse_conn[405,485] = 1

    PLG_adverse_conn[1443,227] = 1
    PLG_adverse_conn[1477,226] = 1
    PLG_adverse_conn[1477,226] = 1

    return PLG_adverse_conn


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


def fast_path_tree_generation(PLG: PLG, start_node: int, target_cluster: int, degree=2, max_path_length=20, max_lane_change=2, min_num_paths=3):
    """Generate a tree of possible paths.

    num_lanes_in_map
                   - The total number of lanes in the map.
    """
    # Initiailise some constants
    num_lanes_in_map = len(set(PLG.node_lane_ids))
    path = [start_node]
    paths = {}

    # Call _fast_path_generation until we generate enough paths, or until we
    # can't call it anymore.
    # - First we constrain the algorithm to generate ONLY paths with less lane
    #   changes than a specified threshold, max_lane_change.
    # - If we cannot generate enough paths to meet our minimum path requirement
    #   increase this max allowed threshold and try again.
    while (len(paths) < min_num_paths) and (max_lane_change <= num_lanes_in_map):
        # Initialisions. Yes - we initialise these two variables above aswell.
        # This is because we need the length in the while-statement so we have
        # to initialise them above. It's a bit ugly but that's ok :(
        path = [start_node]
        paths = {}
        
        # Generate paths
        rc = _fast_path_tree_generation(PLG, target_cluster, path, paths, max_lane_change=max_lane_change, degree=degree, max_path_length=max_path_length)
        max_lane_change += 1

    # We found some paths, now return them
    return paths


def _fast_path_tree_generation(PLG: PLG, target_cluster: int, path: list, paths: dict, degree=2, max_path_length=15, max_lane_change=2):
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
    # that this is the case and if this fails then something went wrong
    # before we tried to call this function. It's not our fault but we can stop
    # the error from propagating any further.
    assert (path[0] - int(path[0])) == 0
    path[0] = int(path[0])

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
    if len_of_path > min_path_length_for_n_lane_changes:
        # There has to be more than max_lane_change+1 nodes in the path for
        # the number of lane changes to exceed max_lane_change. This check
        # saves us calling count_num_lane_changes unnecessarily.
        num_lane_changes = count_num_lane_changes(PLG, path)
    else:
        # If we can't exceed the maximum anyways then just set this value to 0.
        num_lane_changes = 0

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
        if (num_lane_changes <= max_lane_change) and \
           (not last_element_is_none) and \
           (num_unique_nodes == len_of_path) and \
           (not _check_for_jaggy_path(PLG, path)):
            paths[len(paths)] = path

    else:
        # Loop every neighbour and generate a path
        for ii in range(degree):
            # Get the next node
            next_node = arg_max_p_next_node_given_target(PLG.p_next_node_given_target, PLG.p_next_node, closest_clusters_list, path[-1], n_max=ii+1)
            # Recursively call into _fast_path_tree_generation and extend the
            # path by the next_node
            rc = _fast_path_tree_generation(PLG, target_cluster, path+[next_node], paths, max_lane_change=max_lane_change, degree=degree, max_path_length=max_path_length)

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
        node_ii = int(node_list[ii])
        node_jj = int(node_list[ii+1])
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


def node_path_to_output_data(PLG: PLG, node_path: list, mov_avg_win=HEAD_ANG_MOV_AVG_WIN):
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


def plot_node_path(PLG: PLG, node_path: list, color="red", linewidth=1):
    plt.plot(PLG.nodes[node_path, 0], PLG.nodes[node_path, 1], linewidth=linewidth, color=color, zorder=20)
    return True


def scatter_plot_nodes(PLG: PLG, node_path: list, color="red", s=10):
    plt.scatter(PLG.nodes[node_path, 0], PLG.nodes[node_path, 1], s=s, color=color, zorder=20)
    return True


def scatter_vehicles(v_list: list, color="red"):
    """Plot the list of vehicles in v_list. We will use the self.current_state
    of the vehicle for the plot.

    Args:
        v_list (list): List of Vehicle objects.
    """
    for V in v_list:
        # Plot vehicle
        g.plot_rectangle(X=V.get_rectangle(), color=color)


def _solve_quadratic(a: float, b: float, c: float):
    """Solves a quadratic in the form:
      
       a*t^2 + b*t + t = 0

    Note that we are solving for a time, t, in this case. Since this is a
    physical quantity, we're going to disallow complex roots to this quadratic.
    The solution to this quadratic is meant to be a time-to-collision, a
    complex solution simply means that the two vehicles will not collide
    therefore we return an infinite TTC in this case. We'll use the value
    "INF" to represent this case.

    The solution to the quadratic is:

           - b +/- sqrt(b^2 - 4*a*c)
       t = -------------------------
                     2*a              
       
    """
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c

    # Calculate the roots
    if discriminant >= 0:
        root1 = (-b - math.sqrt(discriminant))/(2*a)
        root2 = (-b + math.sqrt(discriminant))/(2*a)
        roots = [root1, root2]
        if min(roots) > 0:
            return min(min(roots), INF)
        else:
            return min(max(roots), INF)
    else:
        return INF


def _calculate_distance_between_nodes(PLG_: PLG, n1: int, n2: int):
    """Calculate the Euclidean distance between two nodes n1 and n2.
    """
    # Coords of node 1
    x1 = PLG_.nodes[n1, 0]
    y1 = PLG_.nodes[n1, 1]
    # Coords of node 2
    x2 = PLG_.nodes[n2, 0]
    y2 = PLG_.nodes[n2, 1]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def _vt_2at2(v: float, a: float, t: float):
    """Calculates: s = v*t + (1/2)*a*t^2

    Args:
        v (float): Speed.
        a (float): Acceleration.
        t (float): Time.
    """
    return v*t + (1/2)*a*t**2


def _calculate_1d_ttc(ds: float, dv: float, da:float):
    """Calculate the TTC for a one-dimensional case in the following form:

        |              # Where the two vehicles are both travelling upwards.
        A  --- v1, a1  # The symbol definitions are:
        |   |          # - ds = distance between the two vehicles.
        |   |          # - dv = v2 - v1. If this quantity is greater than 0 it
        |   | ds       #        means that the two vehicles will collide in a
        |   |          #        finite time. If this quantity is less than 0
        |   |          #        the two vehicles will may never collide.
        B  --- v2, a2  #        Depending on the acceleration difference.
        |              # - da = a2 - a1. If this is greater than 0 it means
                       #        the following vehicle is speeding up faster
                       #        than the preceding vehicle and vice versa.

        To calculate the TTC here, we use the following SUVAT equation:

          s = v*T + (1/2)*a*T^2

        Set the quantities, s, v and a to the differential quantities described
        above to get:

          ds = dv*T + (1/2)*da*T^2

          (1/2)*da*T^2 + dv*T - ds = 0

        Now we have a quadratic equation in the form a*x^2 + bx + c = 0 where:
          a = (1/2)*da
          b = dv
          c = -ds

        NOTE: It is the responsibility of the caller of this function to ensure
        that ds, dv, and da have been assigned correctly.
    """
    # Set quadratic variables
    a = (1/2)*(da)
    b = dv
    c = -ds

    # If da = 0 then set it to a really small non-zero number. This is because
    # we'll get a division by 0 error in the quadratic formula.
    if a == 0:
        a = INF_SMALL

    # Solve the quadratic - we only want one root
    root = _solve_quadratic(a, b, c)

    return root


def calculate_ttc_and_dtc(PLG_: PLG, path_av: list, speed_av: float, acc_av: float, path_bv: list, speed_bv: float, acc_bv: float):
    """Calculate the TTC and DTC between two vehicles with the future
    trajectories and speeds specified. This calculates the time-to-collision
    for the "AV".

    There are 3 unique cases here:

    - Case 1 - AV is leading - 1D:

        |    # In this case both vehicles are travelling upwards along the
        A    # lane. The "AV" is leading. In this case there will be a finite
        |    # TTC if the following vehicle is moving faster than our vehicle.
        |    # We'll represent this using a negative value for the TTC to
        B    # indicate that the risk is due to a fast travelling vehicle
        |    # behind us.

    - Case 2 - AV is following - 1D:

        |    # Both vehicles are travelling upwards. The "AV" is following. We
        B    # have a finite TTC if we're travelling faster than the BV. We 
        |    # represent this case using a finite TTC which is greater than 0.
        |    
        A    
        |    

    - Case 3 - Future intersection point - 2D:

        |  |  # Both vehicles must travel a finite distance to the future
        x  |  # intersection point (denoted by "x"). This is a 2D case because
        |\ |  # it involved a lane change. The collision cases where both
        | \|  # vehicles remain in a single lane throughout the course of the
        |  |  # simulation are 1D because we can thing travelling along a lane
        B  A  # as though being constrained to a single line.
        |  |  # 
              # Calculating the TTC in this case is slightly less straight
              # forward because the standard formula SUVAT formula
              # s = u*t + (1/2)*a*t^2 does not hold between the two vehicles
              # simultaneously.
              #
              # In this case we do the following, calculate the time taken for
              # each vehicle to go from their current location to "x".
              # We simulate the motion of the two vehicles until the first
              # vehicle arrives at point "x". We then treat this as the
              # standard 1D case described above.
    """
    # First check if there is a node in common between the paths of the two
    # vehicles
    nodes_in_common = set(path_av).intersection(set(path_bv))
    if len(nodes_in_common) == 0:
        # If there are no nodes in common then set the ttc and dtc to and
        # infinite value
        ttc = INF
        dtc = INF
        return ttc, dtc
    else:
        # We need to store the first node of each vehicles path. If thist
        # first node is the list of nodes that the paths have in common, then
        # this is a simple 1D collison case. This is either case 1 or case 2.
        current_node_av = path_av[0]
        current_node_bv = path_bv[0]

        if current_node_av in nodes_in_common:
            # Case 1 - The current AV position is in the intersection of the
            # nodes.
            ds = _calculate_distance_between_nodes(PLG_, current_node_av, current_node_bv)
            dv = speed_bv - speed_av
            da = acc_bv - acc_av

            # Calculate the TTC
            ttc = _calculate_1d_ttc(ds, dv, da)
            dtc = -ds

            if ttc == INF:
                return ttc, dtc
            else:
                return -ttc, dtc

        elif current_node_bv in nodes_in_common:
            # Case 2 - The current BV position is in the intersection of the
            # nodes.
            ds = _calculate_distance_between_nodes(PLG_, current_node_av, current_node_bv)
            dv = speed_av - speed_bv
            da = acc_av - acc_bv

            # Calculate the TTC
            ttc = _calculate_1d_ttc(ds, dv, da)
            dtc = ds
            return ttc, dtc

        else:
            # Case 3 - Neither starting node is in the intersection of the
            # paths but the two vehicles paths intersect at some point so this
            # is case 3.
            # 
            # First find the node "x".
            node_x = None
            for node in path_av:
                if node in nodes_in_common:
                    node_x = node
                    break
            assert node_x != None
            assert node_x != current_node_av
            assert node_x != current_node_bv

            # Get distance between vehicles and node_x
            ds_av = _calculate_distance_between_nodes(PLG_, current_node_av, node_x)
            ds_bv = _calculate_distance_between_nodes(PLG_, current_node_bv, node_x)

            # Calculate time for each vehicle to get to node x
            t_av = _calculate_1d_ttc(ds_av, speed_av, acc_av)
            t_bv = _calculate_1d_ttc(ds_bv, speed_bv, acc_bv)

            if t_av < t_bv:
                # AV will get there first
                #
                # Calculate the distance that the BV will have travelled in
                # the time it takes the AV to get to node_x.
                bv_distance_travelled_in_t_av = _vt_2at2(speed_bv, acc_bv, t_av)
                assert bv_distance_travelled_in_t_av < ds_bv

                # Distance left to travel after the AV gets to node x
                ds = ds_bv - bv_distance_travelled_in_t_av

                # Calculate the TTC from this point onwards. We now have a 1D
                # case.
                dv = speed_bv - speed_av
                da = acc_bv - acc_av
                ttc_ = _calculate_1d_ttc(ds, dv, da)
                dtc = -ds


                # The total TTC is then the time taken for the AV to get to
                # node_x plus the time TTC after that.
                if ttc_ == INF:
                    return INF, dtc
                else:
                    return -(t_av + ttc_), dtc
            else:
                # BV will get there first
                #
                # Calculate the distance that the AV will have travelled in
                # the time it takes the BV to get to node_x.
                av_distance_travelled_in_t_bv = _vt_2at2(speed_av, acc_av, t_bv)
                assert av_distance_travelled_in_t_bv < ds_av

                # Distance left to travel after the BV gets to node x
                ds = ds_av - av_distance_travelled_in_t_bv

                # Calculate the TTC from this point onwards. We now have a 1D
                # case.
                dv = speed_av - speed_bv
                da = acc_av - acc_bv
                ttc_ = _calculate_1d_ttc(ds, dv, da)
                dtc = ds

                # The total TTC is then the time taken for the AV to get to
                # node_x plus the time TTC after that.
                if ttc_ == INF:
                    return INF, dtc
                else:
                    return t_bv + ttc_, dtc


def calculate_num_lane_changes(PLG_: PLG, path: list):
    """Calculate the number of lane changes in the path.
    """
    # Initialisations
    num_lane_change = 0

    # Get the length of the path and check that it's atleast length 2 otherwise
    # we can't do anything here.
    path_length = len(path)
    if path_length < 2:
        return 1

    # Now iterate over the path and count the number of lane changes
    for ii in range(path_length-1):
        # Get current lane ID and previous lane ID
        current_lid = PLG_.node_lane_ids[path[ii+1]]
        previous_lid = PLG_.node_lane_ids[path[ii]]
        
        # Check for a lane change
        if current_lid != previous_lid:
            num_lane_change += 1

    return num_lane_change


def count_num_lane_changes(PLG_: PLG, path: list):
    """Counts the number of lane changes in the nodal path specified. Returns
    an integer value.
    """
    # Initialisations
    num_lane_changes = 0
    path_length = len(path)
    lane_ids = [PLG_.node_lane_ids[path[ii]] for ii in range(path_length)]


    # If the length of the path is less than or equal to 1 then there can be no
    # lane changes so return 0
    if path_length <= 1:
        return num_lane_changes

    # Loop through the path and count the lane changes
    for ii in range(path_length-1):
        if lane_ids[ii] != lane_ids[ii+1]:
            num_lane_changes += 1

    return num_lane_changes


def _check_for_jaggy_path(PLG_: PLG, path: list):
    """We want to avoid generating "jaggy" paths where we switch lanes twice
    within 3 nodes. These aren't realistic paths and are a waste of computation
    if we generate many of them. E.g.

    We don't want:    
        o o
        | |
        o o
        |/|
        o o
        |\|
        o o
        | |
        o o

    Returns True if the path is jaggy and False otherwise.
    """
    # Initialisations
    path_length = len(path)

    # This function only applies to pahts which have a length of atleast 3
    # nodes, if this isn't True return False
    if len(path) < 3:
        return False

    # Now iterate over the path and count the number of lane changes
    for ii in range(path_length-2):
        # Get current lane ID and previous lane ID
        lid_0 = PLG_.node_lane_ids[path[ii]]
        lid_1 = PLG_.node_lane_ids[path[ii+1]]
        lid_2 = PLG_.node_lane_ids[path[ii+2]]
        
        # If this path is jaggy
        if (lid_0 != lid_1) and (lid_0 == lid_2) and (lid_1 != lid_2):
            return True

    return False 