import sys
import os

# On my machine I need this line otherwise I get a "ModuleNotFoundError" when
# trying to import the other modules I have written within this directory.
sys.path.append(os.getcwd())

import functions.general as g
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


###############################################################################
# ABOUT THIS SCRIPT:                                                          #
#                                                                             #
# - Generates an initial state for a simulation and plots this state.         #
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

    # Get v_list and AV
    v_list = sim.generate_random_initial_platoon_state(PLG_)
    AV = v_list[0]

    # Generate some paths
    path = graph.path_generation(PLG_, int(AV.current_state.node), AV.target_destination)
    paths = graph.fast_path_tree_generation(PLG_, AV.current_state.node, AV.target_destination, max_path_length=10)
    print(date_time.get_current_time(), "Number of paths generated =", len(paths))

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

