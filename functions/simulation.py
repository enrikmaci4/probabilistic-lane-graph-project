import sys
import os

# On my machine I need this line otherwise I get a "ModuleNotFoundError" when
# trying to import the other modules I have written within this directory.
sys.path.append(os.getcwd())

import copy
import functions.general as g
from functions.general import progressbar
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
def _initialise_current_state(
    PLG_: PLG, start_node: int, target_cluster: int, vehicle_id: int
):
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
    initial_state.speed = random.uniform(SPEED_MEAN - SPEED_STD, SPEED_MEAN + SPEED_STD)
    initial_state.acc = acc_models.linear(ttc=graph.INF, dtc=graph.INF)
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
def initialise_av_position(
    PLG_: PLG, start_node=None, target_cluster=None, initial_node=None
) -> Vehicle:
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
    if start_node == None:
        start_node = int(np.random.choice(PLG_.start_clusters[start_cluster]))
    # We want kind of long paths to get a nice long simulation so choose
    # one of the target clusters which is further away.
    start_coords = PLG_.nodes[start_node, :]
    distance_from_start_to_targets = np.sum(
        np.square(PLG_.target_cluster_centres - start_coords), axis=1
    )
    # TODO: The way we select this could probable improve to get more
    # variation in target clusters. E.g. randomly choose one between the mid-
    # point and the end in a list ordered from lowest to highest (or vice
    # versa).
    if target_cluster == None:
        target_cluster = np.argmax(distance_from_start_to_targets)
    # If you want to set your own values, uncomment the following lines and
    # define them here.
    # start_node = 374
    # target_cluster = 0
    # initial_node = 341
    print(date_time.get_current_time(), f"start_node = {start_node}")
    print(date_time.get_current_time(), f"target_cluster = {target_cluster}")
    print(date_time.get_current_time(), f"initial_node = {initial_node}")

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
    a_low_ind = int(a_low * len(path))
    a_upp_ind = int(a_upp * len(path))
    if initial_node == None:
        initial_node = np.random.choice(path[a_low_ind:a_upp_ind])
    initial_node_index = path.index(initial_node)

    # If you want to set your own values, uncomment the following lines and
    # define them here. NOTE: There is a start_node above which needs to be
    # uncommented as well.
    print(date_time.get_current_time(), f"initial_node = {initial_node}")

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
    initial_state.speed = random.uniform(SPEED_MEAN - SPEED_STD, SPEED_MEAN + SPEED_STD)
    initial_state.acc = acc_models.linear(ttc=graph.INF, dtc=graph.INF)
    initial_state.head_ang = output_data[initial_node_index, 2]

    # Create the AV
    AV = Vehicle(PLG_, initial_state, target_cluster)

    return AV


###############################################################################
# Generate vehicles to surround the AV. I.e. a platoon of vehicles. We are    #
# going to use this platoon to generate traffic simulations.                  #
#                                                                             #
# Params: IN     PLG_   - A PLG object describing the map.                    #
#         IN     AV     - A data structure of type classes.vehicle.Vehicle    #
#                         with an initial state but no further trajectory     #
#                         data.                                               #
#         IN/OUT v_list - A list of Vehicle type data structures. This list   #
#                         describes the platoon of vehicles we're simulating. #
#                         If this list is provided, then the first element    #
#                         should be the AV/ego vehicle.                       #
#                                                                             #
###############################################################################
def generate_platoon(PLG_: PLG, AV: Vehicle, v_list=None):
    # Initialisations
    num_bvs = NUM_BVS
    if v_list == None:
        v_list = [AV]
    else:
        AV = v_list[0]

    # First we're going to get a list of nodes which we know are within the AV
    # detection zone of the AV.
    list_of_nodes_in_detection_zone = []
    for ii in range(AV.PLG.num_nodes):
        if g.is_in_rectangle(
            AV.PLG.nodes[ii, 0],
            AV.PLG.nodes[ii, 1],
            AV.current_state.x,
            AV.current_state.y,
            AV.current_state.head_ang,
            Rx=BV_DETECTION_RX,
            Ry=BV_DETECTION_RY,
        ) and (ii != AV.current_state.node):
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
        initial_state = _initialise_current_state(
            PLG_,
            start_node,
            AV.target_destination,
            AV.current_state.vehicle_id + ii + 1,
        )
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


###############################################################################
# Function used to generate a random initial seed state to begin a            #
# simulation. This function will return a list of vehicle objects. Each       #
# vehicle's state has been intialised and has a trajectory length of 1.       #
#                                                                             #
# Params IN     PLG_   - PLG data structure.                                  #
#        OUT    v_list - A list of Vehicle objects whose state has been       #
#                        initialised.                                         #
#                                                                             #
###############################################################################
def generate_random_initial_platoon_state(PLG_: PLG):
    # Create an AV/ego vehicle
    AV = initialise_av_position(PLG_)

    # Generate BVs randomly around the AV and this will be our v_list
    v_list = generate_platoon(PLG_, AV)

    return v_list


###############################################################################
# Function to generate a single simulation and save it in either the cc or    #
# the ncc directories.                                                        #
#                                                                             #
# Params IN     PLG      - Data structure for PLG.                            #
#        IN     II       - The simulation ID. I.e. if this is the 10th        #
#                          simulation then II will be set to 9 by the while   #
#                          loop which calls this function.                    #
#        IN     MAX_WAIT_TIME                                                 #
#                        - This is the max amount of time we will wait for a  #
#                          simulation to generate. If it has not generated by #
#                          this time then it's taking too long, terminate and #
#                          try again.                                         #
#        IN     SAVE_LOC - Location to save the output data to.               #
#        IN     save_initial_state                                            #
#                        - Set to True if you want to save the initial state  #
#                          as well in a file with a "_is" suffix.             #
#        IN     v_list   - List of vehicles whose state has been initialised. #
#        IN     assert_on_short_sim                                           #
#                        - If the simulations generated are too short, i.e.   #
#                          less than MOV_AVG_WIN, assert and try generating   #
#                          another simulation.                                #
#        IN     generate_only_cc                                              #
#                        - Set to True if you want to generate only crash     #
#                          events. If the output is not a crash event then    #
#                          assert and try again.                              #
#                                                                             #
###############################################################################
def generate_single_simulation(
    PLG_: PLG,
    II="-1",
    MAX_WAIT_TIME=300,
    SAVE_LOC=None,
    save_initial_state=True,
    v_list=None,
    assert_on_short_sim=True,
    generate_only_cc=False,
):
    # Generate a platoon of vehicles if one has not been provided
    if v_list == None:
        v_list = generate_random_initial_platoon_state(PLG_)

    # Create a copy of the initial state
    v_list_is = copy.deepcopy(v_list)

    # Simulation params
    sim_frame_length = int(round(SIM_LENGTH / dt, 0))
    num_vehicles = len(v_list)
    terminate_simulation = False
    is_cc = False

    # Simulation
    t_start = time.time()
    for ii in progressbar(range(sim_frame_length), prefix=f"Simulating: {II} "):
        # For each time step loop over every vehicle and take a "step" i.e.
        # update it's state over a period "dt".
        for jj in range(num_vehicles):
            # Simulate a time step for this vehicle
            rc = v_list[jj].step(ii, v_list)

            # Check if we should terminate the simulation
            if rc == SIGNAL_TERM_SIM:
                terminate_simulation = True

            # If this is taking too long then break
            if time.time() - t_start > MAX_WAIT_TIME:
                terminate_simulation = True

        # Now check for collisions and print log
        if g.check_for_collision(v_list, store_collision=True):
            rc = SIGNAL_COLLISION
            is_cc = True
            terminate_simulation = True

        # We need to break out of the outer loop too
        if terminate_simulation:
            break
    # Print an empty line so print statements after loading bar format
    # correctly
    print()

    # Assert a minimum path length otherwise this cc was probably a
    # result of some bad initial state
    if assert_on_short_sim:
        assert v_list[0].trajectory_length >= MOV_AVG_WIN

    # Store the time taken so we can return it and calculate average time taken
    time_taken = time.time() - t_start

    # Smooth the x, y and heading angle columns
    for V in v_list:
        rc = g.smooth_output_data(V, mov_avg_win=MOV_AVG_WIN, keep_end=True)

    # Save data
    if is_cc:
        g.save_pickled_data(SAVE_LOC + SIM_DATA_PKL_NAME + "_" + II + CC_SUFF, v_list)
    else:
        if generate_only_cc:
            # If this boolean is set to True, we only want to generate crash
            # cases. If this isn'y a crash case then assert so that we don't
            # save the case and because of the assertion the progream will
            # retry generating this simulation.
            assert is_cc == True
        g.save_pickled_data(SAVE_LOC + SIM_DATA_PKL_NAME + "_" + II + NCC_SUFF, v_list)

    # There were no errors, save the initial state too
    if save_initial_state:
        g.save_pickled_data(
            SAVE_LOC + SIM_DATA_PKL_NAME + "_" + II + IS_SUFF, v_list_is
        )

    return is_cc
