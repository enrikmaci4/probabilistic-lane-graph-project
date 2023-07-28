import sys
import os

# On my machine I need this line otherwise I get a "ModuleNotFoundError" when
# trying to import the other modules I have written within this directory.
sys.path.append(os.getcwd())

import functions.general as g
import functions.date_time as date_time
import functions.graph as graph
import time
import matplotlib.pyplot as plt
import random
from inputs import *
from fnames import *
import numpy as np
from classes.PLG import *
from classes.vehicle import *
import models.acceleration as acc_models


###############################################################################
# ABOUT THIS SCRIPT:                                                          #
#                                                                             #
# - Generates an initial state for a simulation and plots this state.         #
#                                                                             #
###############################################################################

###############################################################################
# Function to intialise the self.current_state of a vehicle given a start     #
# node and target destination. The target destination is one of the target    #
# clusters. We initialise the current_state by generating a path from our     #
# current location to the target destination and taking the average heading   #
# angle from there. It's not ideal but it'll work.                            #
#                                                                             #
# Params: PLG_       - A PLG object describing the map.                       #
#         start_node - An integer from 0 to number of nodes in the PLG-1.     #
#                      This describes the starting location of the vehicle    #
#                      whose current_state we would like to intialise.        #
#         target_cluster                                                      #
#                    - An integer from 0 to number of target clusters-1. This #
#                      is the target destination as a cluster of nodes.       #
#         vehicle_id - Integer. The ID of this vehicle.                       #
#                                                                             #
# Returns:                                                                    #
#         - A DataRow object which describes the initial state of this        #
#           vehicle.                                                          #
#                                                                             #
###############################################################################
def initialise_current_state(PLG_: PLG, start_node: int, target_cluster: int, vehicle_id: int):
    # Generate a path
    path = graph.path_generation(PLG_, start_node, target_cluster)

    # Get an "output_data" data structure. This data structure converts a node
    # path into a 2D matrix with columns [x, y, heading angle]. We're mainly
    # interested in the heading angle.
    output_data = graph.node_path_to_output_data(PLG_, path)

    # Create a DataRow data structure and populate it with the data we need to
    # initialise the AV.
    initial_node_index = 0
    initial_node = start_node

    initial_state = DataRow()
    initial_state.vehicle_id = vehicle_id
    initial_state.time = vehicle_id
    initial_state.x = output_data[initial_node_index, 0]
    initial_state.y = output_data[initial_node_index, 1]
    initial_state.node = initial_node
    initial_state.lane_id = PLG_.node_lane_ids[initial_node]
    initial_state.speed = random.uniform(PLG_.statistics.speed_min, PLG_.statistics.speed_max)
    initial_state.acc = acc_models.linear(graph.INF, A_max=PLG_.statistics.acc_max)
    initial_state.head_ang = output_data[initial_node_index, 2]

    return initial_state


###############################################################################
# Use this function to generate the initial vehicle position. This will be    #
# our "AV". However, all vehicles will run under the same policy in this test #
# so the term "AV" is really just synonymous with the "subject vehicle", i.e. #
# the one that we will focus on when we generate our visualisations.          #
#                                                                             #
# Returns: av - A "Vehicle" type object with an initial state.                #
#                                                                             #
###############################################################################
def initialise_av_position(PLG_: PLG) -> Vehicle:
    # First we're going to generate a single vehicle path from an entry point
    # to an exit. We will then choose a random node, somewhere in the middle
    # of the path, and choose this as the initial state of the AV. There are a
    # few reasons for this:
    # - The entry and exit locations can be a bit funky and we'd rather
    #   generate cleaner simulations for now, atleast until we understand the
    #   model better.
    # - We don't really want vehicle's to suddenly disappear from the map which
    #   is what will happen if a vehicle reaches it's target destination.
    start_cluster = int(np.random.choice(list(PLG_.start_clusters.keys())))
    start_node = int(np.random.choice(PLG_.start_clusters[start_cluster]))
    # We want kind of long paths to get a nice long simulation so choose 
    # one of the target clusters which is further away.
    start_coords = PLG_.nodes[start_node,:]
    distance_from_start_to_targets = np.sum(np.square(PLG_.target_cluster_centres - start_coords), axis=1)
    # TODO: The way we select this could probable improve to get more
    # variation in target clusters. E.g. randomly choose one betwee the mid-
    # point and the end in a list ordered from lowest to highest (or vice
    # versa).
    target_cluster = np.argmax(distance_from_start_to_targets)

    # Generate a path
    path = graph.path_generation(PLG_, start_node, target_cluster)
    print(date_time.get_current_time(), "Length of path =", len(path))

    # Now choose a random node between the a_low and a_upp percentiles. This
    # is because we want to initialise the AV somewhere in the middle of it's
    # path.
    # NOTE: We aren't using initialise_current_state to generate the initial
    # state of the AV because it's a special case where we already had the
    # previous nodes in the path so we use those to get the heading angle. If
    # we use initialise_current_state then we are approximating the current
    # heading angle as the heading angle across the next few nodes. It's not a
    # bad approximation but we avoid it anyway.
    a_low = 0.25
    a_upp = 0.75
    a_low_ind = int(a_low*len(path))
    a_upp_ind = int(a_upp*len(path))
    initial_node = np.random.choice(path[a_low_ind:a_upp_ind])
    initial_node_index = path.index(initial_node)
    path_trunc = graph.path_generation(PLG_, initial_node, target_cluster, max_path_length=15)

    # Get an "output_data" data structure. This data structure converts a node
    # path into a 2D matrix with columns [x, y, heading angle]. We're mainly
    # interested in the heading angle.
    output_data = graph.node_path_to_output_data(PLG_, path)

    # Create a DataRow data structure and populate it with the data we need to
    # initialise the AV.
    initial_state = DataRow()
    initial_state.vehicle_id = 0
    initial_state.time = 0
    initial_state.x = output_data[initial_node_index, 0]
    initial_state.y = output_data[initial_node_index, 1]
    initial_state.node = initial_node
    initial_state.lane_id = PLG_.node_lane_ids[initial_node]
    initial_state.speed = random.uniform(PLG_.statistics.speed_min, PLG_.statistics.speed_max)
    initial_state.acc = acc_models.linear(graph.INF, A_max=PLG_.statistics.acc_max)
    initial_state.head_ang = output_data[initial_node_index, 2]

    # Create the AV
    AV = Vehicle(PLG_, initial_state, target_cluster) 

    return AV


###############################################################################
# Generate vehicles to surround the AV. I.e. a platoon of vehicles. We are    #
# going to use this platoon to generate traffic simulations.                  #
#                                                                             #
# Params: IN  PLG_   - A PLG object describing the map.                       #
#         IN  AV     - A data structure of type classes.vehicle.Vehicle with  #
#                      an initial state but no further trajectory data.       #
#         OUT v_list - A list of Vehicle type data structures. This list      #
#                      describes the platoon of vehicles we're simulating.    #
#                                                                             #
###############################################################################
def generate_platoon(PLG_:PLG, AV: Vehicle):
    # Initialisations
    num_bvs = 10
    v_list = [AV]

    # First we're going to get a list of nodes which we know are within the AV
    # detection zone of the AV.
    list_of_nodes_in_detection_zone = []
    for ii in range(AV.PLG.num_nodes):
        if g.is_in_rectangle(AV.PLG.nodes[ii,0],
                             AV.PLG.nodes[ii,1],
                             AV.current_state.x,
                             AV.current_state.y,
                             AV.current_state.head_ang,
                             Rx=BV_DETECTION_RX,
                             Ry=BV_DETECTION_RY) and \
           (ii != AV.current_state.node):
            # This node is in the detection zone and it is not the AV's current
            # node so add it to the list of available nodes
            list_of_nodes_in_detection_zone.append(ii)

    # Now create a list of start nodes for each BV
    bv_start_nodes = []
    ii = 0
    while (ii < num_bvs) and (len(list_of_nodes_in_detection_zone) > 0): 
        # Randomly choose a start node
        start_node = np.random.choice(list_of_nodes_in_detection_zone)
        # Add this node to bv_start nodes and remove it from
        # list_of_nodes_in_detection_zone so that we don't get it again
        list_of_nodes_in_detection_zone.remove(start_node)

        # Create a BV on this node
        initial_state = initialise_current_state(PLG_, start_node, AV.target_destination, AV.current_state.vehicle_id+ii+1)
        BV = Vehicle(PLG_, initial_state, AV.target_destination)

        # Only add this node to our list of nodes for BVs if there is no
        # collision with a vehicle on this node and every other vehicle on the
        # map
        if not g.check_for_collision(v_list + [BV], x_scale=3):
            if start_node not in bv_start_nodes:
                bv_start_nodes.append(start_node)
                v_list.append(BV)

        # Increment counter
        ii += 1

    return v_list


def main():
    # Time the script
    t_start = time.time()
    print(date_time.get_current_time(), "Program started")

    # Load the cleaned data
    data = g.load_pickled_data(CLEAN_DATA_LOC+CLEAN_DATA_NAME)
    print(date_time.get_current_time(), "Loaded clean data")

    # Create a PLG object
    PLG_ = g.load_pickled_data(PLG_SAVE_LOC+PLG_NAME)
    print(date_time.get_current_time(), "Loaded PLG")

    # Print time take
    print(f"{date_time.get_current_time()} Time taken to load data = {round(time.time() - t_start, 3)} s")

    # Initialise an AV
    AV = initialise_av_position(PLG_)

    # Generate some paths
    path = graph.path_generation(PLG_, int(AV.current_state.node), AV.target_destination)
    paths = graph.fast_path_tree_generation(PLG_, AV.current_state.node, AV.target_destination, max_path_length=10)
    print(date_time.get_current_time(), "Number of paths generated =", len(paths))

    # Generate a platoon of vehicles
    v_list = generate_platoon(PLG_, AV)

    # Do some plots
    graph.draw(PLG_)
    ii_random = random.randint(0, len(paths)-1)
    for ii in paths:
        graph.plot_node_path(PLG_, paths[ii], color="orange")

    graph.plot_node_path(PLG_, path)
    graph.plot_node_path(PLG_, paths[ii_random], color="yellow")
    graph.scatter_vehicles(v_list, color="red")
    g.plot_rectangle(AV.get_rectangle(), color="skyblue")
    g.plot_rectangle(xc=AV.current_state.x, yc=AV.current_state.y, Rx=BV_DETECTION_RX, Ry=BV_DETECTION_RY, alpha=AV.current_state.head_ang, color="grey")    

    # Set the aspect ratio to be equal
    plt.xlim([AV.current_state.x-SCREEN_WIDTH/2, AV.current_state.x+SCREEN_WIDTH/2])
    plt.ylim([AV.current_state.y-SCREEN_HEIGHT/2, AV.current_state.y+SCREEN_HEIGHT/2])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


if __name__=="__main__":
    main()

