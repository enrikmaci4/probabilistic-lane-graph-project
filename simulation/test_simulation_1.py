import sys
import os

# On my machine I need this line otherwise I get a "ModuleNotFoundError" when
# trying to import the other modules I have written within this directory.
sys.path.append(os.getcwd())

import functions.general as g
from functions.general import progressbar
import functions.date_time as date_time
import functions.graph as graph
import time
import matplotlib.pyplot as plt
import random
from inputs import *
import numpy as np
from classes.PLG import *
from classes.vehicle import *
import models.acceleration as acc_models
#from tqdm import tqdm


DATA_LOC = "data/"+DATASET+"/cleaned/"
PLG_SAVE_LOC = "data/"+DATASET+"/data-structures/"
SIM_DATA_SAVE_LOC = "output-data/simulation/"

###############################################################################
# ABOUT THIS SCRIPT:                                                          #
#                                                                             #
# - Simulate a single scenario and save the data in the output directory. The #
#   data is saved as Python pickle. The data is a list of vehicle structures  #
#   each with the same "trajectory" length.                                   #
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
    # - Some constants used to initialise the state
    speed_mean = 10
    speed_std = 3

    # - Initialise state
    initial_node_index = 0
    initial_node = start_node

    initial_state = DataRow()
    initial_state.vehicle_id = vehicle_id
    initial_state.time = vehicle_id
    initial_state.x = output_data[initial_node_index, 0]
    initial_state.y = output_data[initial_node_index, 1]
    initial_state.node = initial_node
    initial_state.lane_id = PLG_.node_lane_ids[initial_node]
    initial_state.speed = random.uniform(speed_mean-speed_std, speed_mean+speed_std)
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
    # - Some constants used to initialise the state
    speed_mean = 15
    speed_std = 3

    # - Initialise state
    initial_state = DataRow()
    initial_state.vehicle_id = 0
    initial_state.time = 0
    initial_state.x = output_data[initial_node_index, 0]
    initial_state.y = output_data[initial_node_index, 1]
    initial_state.node = initial_node
    initial_state.lane_id = PLG_.node_lane_ids[initial_node]
    initial_state.speed = random.uniform(speed_mean-speed_std, speed_mean+speed_std)
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
    num_bvs = 5
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
    data = g.load_pickled_data(DATA_LOC+"clean_data_v2")
    print(date_time.get_current_time(), "Loaded clean data")

    # Create a PLG object
    PLG_ = g.load_pickled_data(PLG_SAVE_LOC+"PLG")
    print(date_time.get_current_time(), "Loaded PLG")

    # Print time take
    print(f"{date_time.get_current_time()} Time taken to load data = {round(time.time() - t_start, 3)} s")

    # Initialise an AV
    AV = initialise_av_position(PLG_)

    # Generate a platoon of vehicles
    v_list = generate_platoon(PLG_, AV)
    print(date_time.get_current_time(), f"Generated platoon with {len(v_list)} vehicles")

    # Save platoon incase we want to re-use it
    load_platoon = False
    if load_platoon:
        v_list = g.load_pickled_data(SIM_DATA_SAVE_LOC+"platoon")
    else:
        g.save_pickled_data(SIM_DATA_SAVE_LOC+"platoon", v_list)

    #print(v_list[0].trajectory)
    #g.smooth_output_data(v_list[0])
    #quit()
    
    # Simulation params
    sim_frame_length = int(round(SIM_LENGTH/dt, 0))
    data = np.zeros((0, NUM_COLS_IN_DATA_MATRIX))
    num_vehicles = len(v_list)
    terminate_simulation = False

    # Simulation
    t_ = time.time()
    for ii in progressbar(range(sim_frame_length), prefix="Simulating: "):
        # For each time step loop over every vehicle and take a "step" i.e.
        # update it's state over a period "dt".
        for jj in range(num_vehicles):
            # Simulate a time step for this vehicle
            rc = v_list[jj].step(ii, v_list)

            # Check if we should terminate the simulation
            if rc == SIGNAL_TERM_SIM:
                terminate_simulation = True

        # Now check for collisions
        if g.check_for_collision(v_list, store_collision=True):
            terminate_simulation = True
            print(date_time.get_current_time(), "!!! Vehicles collided", "")

        # We need to break out of the outer loop too
        if terminate_simulation:
            print(date_time.get_current_time(), "Terminating simulation! Either a collision occurred or a vehicle reached it's target destination.")
            break

    print(date_time.get_current_time(), "Time taken =", round(time.time()-t_, 3))

    # Smooth the x, y and heading angle columns
    for V in v_list:
        rc = g.smooth_output_data(V, mov_avg_win=10, keep_end=True)

    # TODO: Sometimes this script fails. Will fix...
    # TODO: Sometimes we get weird heading angle stuff where the vehicle
    #       rotates. This is because we're taking the moving average of the
    #       heading angle but we need to find a way to stop this weird
    #       rotations. They are rare but still need to prevented...
    
    # Get a data matrix
    for ii in range(num_vehicles):
        data = np.vstack((data, v_list[ii].trajectory))

    # Save data
    np.savetxt(SIM_DATA_SAVE_LOC+"test_data", data)
    g.save_pickled_data(SIM_DATA_SAVE_LOC+"test_list", v_list)


if __name__=="__main__":
    main()

