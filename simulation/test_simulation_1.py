import sys
import os

# On my machine I need this line otherwise I get a "ModuleNotFoundError" when
# trying to import the other modules I have written within this directory.
sys.path.append(os.getcwd())

import functions.general as g
from functions.general import progressbar
import functions.date_time as date_time
import functions.graph as graph
import functions.simulation as sim
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
# - Simulate a single scenario and save the data in the output directory. The #
#   data is saved as Python pickle. The data is a list of vehicle structures  #
#   each with the same "trajectory" length.                                   #
# - The simulation data will be saved in TEST_SIM_SAVE_LOC.                   #
#                                                                             #
###############################################################################

# If you would like to load the IS which is located in output/test then set
#
#   LOAD_PLATOON = True
#   II = None
#
# Otherwise, if you would like to load an output from output/set1 then set this
# variable to the index of the initial state you would like to load. I.e. if
# you would like to load output/set1/simdata_pkl_10_is then set:
#
#   LOAD_PLATOON = True
#   II = 10
#
LOAD_PLATOON = False
II = None

# Initialise the location to load data from
if II == None:
    LOAD_DATA_LOC = f"{TEST_SIM_SAVE_LOC}{SIM_DATA_PKL_NAME}{IS_SUFF}"
else:
    LOAD_DATA_LOC = f"{SET1_SAVE_LOC}{SIM_DATA_PKL_NAME}_{II}{IS_SUFF}"

# Create a variable to store the save lcoation of the simulation.
SAVE_DATA_LOC = f"{TEST_SIM_SAVE_LOC}{SIM_DATA_PKL_NAME}"


def main():
    # Time the script
    t_start = time.time()
    print(date_time.get_current_time(), "Program started")

    # Create a PLG object
    PLG_ = g.load_pickled_data(PLG_SAVE_LOC + PLG_NAME)
    print(date_time.get_current_time(), "Loaded PLG")

    # Print time take
    print(
        f"{date_time.get_current_time()} Time taken to load data = {round(time.time() - t_start, 3)} s"
    )

    # Generate a platoon of vehicles
    v_list = sim.generate_random_initial_platoon_state(PLG_)
    print(
        date_time.get_current_time(), f"Generated platoon with {len(v_list)} vehicles"
    )

    # Save platoon incase we want to re-use it
    load_platoon = LOAD_PLATOON
    print(date_time.get_current_time(), f"Save file will be: {SAVE_DATA_LOC}")
    if load_platoon:
        # Load vehicle list
        v_list = g.load_pickled_data(LOAD_DATA_LOC)
    else:
        # Save this initial state incase we want to use it again
        g.save_pickled_data(SAVE_DATA_LOC + IS_SUFF, v_list)

    # Simulation params
    sim_frame_length = int(round(SIM_LENGTH / dt, 0))
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

        # Now check for collisions and print log
        if g.check_for_collision(v_list, store_collision=True):
            rc = SIGNAL_COLLISION
            terminate_simulation = True
            # Print a blank line so that the loading bar doesn't overwrite our
            # log then print the log
            print()
            print(date_time.get_current_time(), "!!! VEHICLES COLLIDED", "")

        if rc == SIGNAL_TERM_SIM:
            # Print a blank line so that the loading bar doesn't overwrite our
            # log then print the log
            print()
            print(date_time.get_current_time(), "!!! TARGET DESTINATION REACHED", "")

        # We need to break out of the outer loop too
        if terminate_simulation:
            # If this is NOT a collision then it is
            print(date_time.get_current_time(), "Terminating simulation")
            break

    print(date_time.get_current_time(), "Time taken =", round(time.time() - t_, 3))

    # Smooth the x, y and heading angle columns
    for V in v_list:
        rc = g.smooth_output_data(V, mov_avg_win=MOV_AVG_WIN, keep_end=True)

    # TODO: Sometimes this script fails. Will fix...

    # Save data
    g.save_pickled_data(SAVE_DATA_LOC, v_list)


if __name__ == "__main__":
    main()
