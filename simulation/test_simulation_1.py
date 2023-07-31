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
#                                                                             #
###############################################################################
def main():
    # Time the script
    t_start = time.time()
    print(date_time.get_current_time(), "Program started")

    # Create a PLG object
    PLG_ = g.load_pickled_data(PLG_SAVE_LOC+PLG_NAME)
    print(date_time.get_current_time(), "Loaded PLG")

    # Print time take
    print(f"{date_time.get_current_time()} Time taken to load data = {round(time.time() - t_start, 3)} s")

    # Generate a platoon of vehicles
    v_list = sim.generate_random_initial_platoon_state(PLG_)
    print(date_time.get_current_time(), f"Generated platoon with {len(v_list)} vehicles")

    # Save platoon incase we want to re-use it
    load_platoon = False
    save_loc_name = TEST_SIM_SAVE_LOC+SIM_DATA_PKL_NAME+IS_SUFF
    if load_platoon:
        # Load vehicle list
        v_list = g.load_pickled_data(save_loc_name)
    else:
        # Save his initial state incase we want to use it again
        g.save_pickled_data(save_loc_name, v_list)
    
    # Simulation params
    sim_frame_length = int(round(SIM_LENGTH/dt, 0))
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

    print(date_time.get_current_time(), "Time taken =", round(time.time()-t_, 3))

    # Smooth the x, y and heading angle columns
    for V in v_list:
        rc = g.smooth_output_data(V, mov_avg_win=20, keep_end=True)

    # TODO: Sometimes this script fails. Will fix...

    # Save data
    g.save_pickled_data(TEST_SIM_SAVE_LOC+SIM_DATA_PKL_NAME, v_list)


if __name__=="__main__":
    main()

