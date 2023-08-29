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
# - Plot's a simulation as a figure using Matplotlib.                         #
#                                                                             #
###############################################################################

# Define which simulation to plot using
# 
#   SET = TEST_SIM_SAVE_LOC or SETi_SAVE_LOC
#   II = None or integer. None for TEST folder and integer for SET folder
# 
SET = SET1_SAVE_LOC
II = 120

if II == None:
    LOAD_DATA_LOC = f"{TEST_SIM_SAVE_LOC}{SIM_DATA_PKL_NAME}"
else:
    LOAD_DATA_LOC = f"{SET}{SIM_DATA_PKL_NAME}_{II}"

# If you want to highlight any vehicle paths, insert them into this list
highlight_ids = []

def main():
    # Time the script
    t_start = time.time()
    print(date_time.get_current_time(), "Program started")

    # Create a PLG object
    PLG_ = g.load_pickled_data(PLG_SAVE_LOC+PLG_NAME)
    print(date_time.get_current_time(), "Loaded PLG")

    # Print time take
    print(f"{date_time.get_current_time()} Time taken to load data = {round(time.time() - t_start, 3)} s")

    # Load scenario
    try:
        v_list = g.load_pickled_data(f"{LOAD_DATA_LOC}{NCC_SUFF}")
    except FileNotFoundError:
        try:
            v_list = g.load_pickled_data(f"{LOAD_DATA_LOC}{CC_SUFF}")
        except FileNotFoundError:
            v_list = g.load_pickled_data(f"{LOAD_DATA_LOC}")

    # PLOTS
    # Plot PLG
    graph.draw(PLG_)

    # Cycle through v_list
    dx = 1
    dy = 1
    for V in v_list:
        # Get vehicle data
        x = V.trajectory[-1, II_X]
        y = V.trajectory[-1, II_Y]
        id = V.current_state.vehicle_id
        x_path = V.trajectory[:, II_X]
        y_path = V.trajectory[:, II_Y]
        if V.is_collision:
            v_color = "orange"
            path_color = "red"
            z_order = 36
        else:
            v_color = "red"
            path_color = "aqua"
            z_order = 35
        if V.current_state.vehicle_id in highlight_ids:
            v_color = "orange"
            path_color = "yellow"
            z_order = 36
        # Plot the vehicle
        g.plot_rectangle(X=V.get_rectangle(-1), color=v_color, plot_heading=True, z_order=37)
        # Plot vehicle path
        plt.plot(x_path, y_path, linewidth=2, color=path_color, linestyle="--", zorder=z_order)
        # Plot vehicle ID
        annot_string = f"id={id}"
        plt.annotate(annot_string, (x+dx, y+dy), size=7.5, fontweight="bold", zorder=50, color="indigo")



    X_CENTRE = v_list[0].trajectory[-1, II_X]
    Y_CENTRE = v_list[0].trajectory[-1, II_Y]
    plt.xlim([X_CENTRE - SCREEN_WIDTH/2, X_CENTRE + SCREEN_WIDTH/2])
    plt.ylim([Y_CENTRE - SCREEN_HEIGHT/2, Y_CENTRE + SCREEN_HEIGHT/2])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.show()


if __name__=="__main__":
    main()

