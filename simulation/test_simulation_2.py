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


NUM_SIMULATIONS = 50


###############################################################################
# ABOUT THIS SCRIPT:                                                          #
#                                                                             #
# - Generate and save many simulations and save them according to whether     #
#   they terminated in a collision or not.                                    #
#                                                                             #
###############################################################################
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

    # Generate a platoon of vehicles
    v_list = sim.generate_random_initial_platoon_state(PLG_)
    print(date_time.get_current_time(), f"Generated platoon with {len(v_list)} vehicles")

    # Save platoon incase we want to re-use it
    load_platoon = False
    name_suffix = "is" # is = initial state
    if load_platoon:
        # Load vehicle list
        v_list = g.load_pickled_data(TEST_SIM_SAVE_LOC+SIM_DATA_PKL_NAME+name_suffix)
    else:
        # Save his initial state incase we want to use it again
        g.save_pickled_data(TEST_SIM_SAVE_LOC+SIM_DATA_PKL_NAME+name_suffix, v_list)
    
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

        # Now check for collisions
        if g.check_for_collision(v_list, store_collision=True):
            terminate_simulation = True
            # Print a blank line then print the collision log
            print()
            print(date_time.get_current_time(), "!!! VEHICLES COLLIDED", "")

        # We need to break out of the outer loop too
        if terminate_simulation:
            print(date_time.get_current_time(), "Terminating simulation! Either a collision occurred or a vehicle reached it's target destination.")
            break

    print(date_time.get_current_time(), "Time taken =", round(time.time()-t_, 3))

    # Smooth the x, y and heading angle columns
    for V in v_list:
        rc = g.smooth_output_data(V, mov_avg_win=20, keep_end=False)
        # TODO: add some more smoothing to the final elements when keep_end is set to TRUE so it's not so jumpy

    # TODO: Sometimes this script fails. Will fix...
    # TODO: Sometimes we get weird heading angle stuff where the vehicle
    #       rotates. This is because we're taking the moving average of the
    #       heading angle but we need to find a way to stop this weird
    #       rotations. They are rare but still need to prevented...
    # TODO: Add termination signal + collision signal

    # Save data
    g.save_pickled_data(TEST_SIM_SAVE_LOC+SIM_DATA_PKL_NAME, v_list)


if __name__=="__main__":
    main()

