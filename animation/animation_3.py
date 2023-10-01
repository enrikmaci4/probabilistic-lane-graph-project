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
from matplotlib.animation import FuncAnimation
from inputs import *
from fnames import *
import numpy as np
from classes.PLG import *
from classes.vehicle import *
from functions.general import progressbar_anim


def load_v_list(fpath):
    try:
        v_list = g.load_pickled_data(f"{fpath}{NCC_SUFF}")
    except FileNotFoundError:
        try:
            v_list = g.load_pickled_data(f"{fpath}{CC_SUFF}")
        except FileNotFoundError:
            v_list = g.load_pickled_data(f"{fpath}")
    return v_list


NEWLINE_CHAR = "\n"
EMPTY_VALUE_STR = "---"


###############################################################################
# INPUT: Define which simulation to plot using                                #
#                                                                             #
#   II = None or integer. None for TEST folder and integer for SET folder     #
#                                                                             #
###############################################################################
II = 14

LOAD_DATA_LOC_1 = f"{SET2_SAVE_LOC}{SIM_DATA_PKL_NAME}_{II}"
LOAD_DATA_LOC_2 = f"{SET1_SAVE_LOC}{SIM_DATA_PKL_NAME}_{II}"


###############################################################################
# This script is used to generate an animation with two simulations starting  #
# from the same initial state.                                                #
#                                                                             #
# SET1 will be used to store the crash events and SET2 will simulate the same #
# scenarios but will not terminate with a crash event. However, these can be  #
# changed. Set LOAD_DATA_LOC_1 to contain the simulation with NO crash and    #
# LOAD_DATA_LOC_2 to contain the simulation WITH a crash event. In my case    #
# this was SET1 and SET2 respectively, however, if this is not the case then  #
# change the two variables above accordingly.                                 #
#                                                                             #
# THis script will take the outputs from SET1 and SET2 and create an          #
# animation putting the two plots side by side to allow for direct            #
# comparison.                                                                 #
#                                                                             #
###############################################################################

# ANIMATION CONSTANTS
START_ANIMATION = False
# Recommended FPS for smooth animations: 10
FPS = 10
FREEZE_FOR_X_SECONDS = 3
dx = 0.75
dy = 0.1
CENTRE_V_ID = 0
# GLOBAL VARIABLES NEEDED FOR ANIMATION
# - PLG object
PLG_ = g.load_pickled_data(PLG_SAVE_LOC + PLG_NAME)
# - Load vehicle data + simulation length
# - - Vehicle 1
v_list_1 = load_v_list(LOAD_DATA_LOC_1)
len_of_sim_1 = v_list_1[0].trajectory_length
v_plot_1 = []
annotation_plot_1 = []
# - - Vehicle 2
v_list_2 = load_v_list(LOAD_DATA_LOC_2)
len_of_sim_2 = v_list_2[0].trajectory_length
v_plot_2 = []
annotation_plot_2 = []
# - Maximum length of simulaiton
len_of_sim = max(len_of_sim_1, len_of_sim_2)
T_MAX = SIM_LENGTH
T_MAX_1 = (len_of_sim_1 / len_of_sim) * T_MAX
T_MAX_2 = (len_of_sim_2 / len_of_sim) * T_MAX


# Initializing a figure in which the graph will be plotted
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_aspect("equal", adjustable="box")
ax2.set_aspect("equal", adjustable="box")
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)


# Animation function
def animate(ii):
    # Global vars
    global START_ANIMATION, PLG_, v_list_1, v_list_2, len_of_sim, len_of_sim_1, len_of_sim_2, ax1, ax2

    if not START_ANIMATION:
        START_ANIMATION = True

    elif ii < len_of_sim:
        # Before plotting, remove the previous plots
        # V1
        if ii < len_of_sim_1:
            num_plots_11 = len(v_plot_1)
            for nn in range(num_plots_11):
                # Start indexing from the final element because we're going to
                # remove
                jj = num_plots_11 - 1 - nn
                # Get the matplotlib objects
                v_plot_11 = v_plot_1[jj]
                annotation_plot_11 = annotation_plot_1[jj]
                # Delete plot
                v_plot_11[0].remove()
                annotation_plot_11.remove()
                # Pop from list
                v_plot_1.pop(jj)
                annotation_plot_1.pop(jj)

        # V2
        if ii < len_of_sim_2:
            num_plots_22 = len(v_plot_2)
            for nn in range(num_plots_22):
                # Start indexing from the final element because we're going to
                # remove
                jj = num_plots_22 - 1 - nn
                # Get the matplotlib objects
                v_plot_22 = v_plot_2[jj]
                annotation_plot_22 = annotation_plot_2[jj]
                # Delete plot
                v_plot_22[0].remove()
                annotation_plot_22.remove()
                # Pop from list
                v_plot_2.pop(jj)
                annotation_plot_2.pop(jj)

        # Cycle through vehicle list and plot the vehicle
        # V1
        if ii < len_of_sim_1:
            for V in v_list_1:
                num_dp = 1
                # Get some constants
                id = int(V.trajectory[ii, II_VEHICLE_ID])
                x = V.trajectory[ii, II_X]
                y = V.trajectory[ii, II_Y]
                speed = round(V.trajectory[ii, II_SPEED], num_dp)
                acc = round(V.trajectory[ii, II_ACC], num_dp)
                ttc = round(V.trajectory[ii, II_TTC], num_dp)
                dtc = round(V.trajectory[ii, II_DTC], num_dp)
                if ttc == graph.INF:
                    ttc = EMPTY_VALUE_STR
                if dtc == graph.INF:
                    dtc = EMPTY_VALUE_STR

                # Plot this vehicle
                v_plot_1.append(
                    g.plot_rectangle(
                        X=V.get_rectangle(ii), color="red", plot_heading=True, axis=ax1
                    )
                )

                # Plot annotations
                annot_color = "indigo"
                annot_string = rf"id={id}"
                if V.is_collision:
                    annot_color = "orange"

                # Plot the IDs next to the vehicles
                annotation_plot_1.append(
                    ax1.annotate(
                        annot_string,
                        (x + dx, y + dx),
                        size=6.5,
                        fontweight="bold",
                        zorder=20,
                        color=annot_color,
                    )
                )

                # Set the axes
                if V.current_state.vehicle_id == CENTRE_V_ID:
                    ax1.set_xlim([x - SCREEN_WIDTH / 2, x + SCREEN_WIDTH / 2])
                    ax1.set_ylim([y - SCREEN_HEIGHT / 2, y + SCREEN_HEIGHT / 2])

                    # Get x,y coordinates of the current time and plot it in the
                    # top right side of the screen
                    T = round(((ii + 1) / len_of_sim_1) * T_MAX_1, 1)
                    ax1.set_title(f"T={T}s")

        # V1
        if ii < len_of_sim_2:
            for V in v_list_2:
                num_dp = 1
                # Get some constants
                id = int(V.trajectory[ii, II_VEHICLE_ID])
                x = V.trajectory[ii, II_X]
                y = V.trajectory[ii, II_Y]
                speed = round(V.trajectory[ii, II_SPEED], num_dp)
                acc = round(V.trajectory[ii, II_ACC], num_dp)
                ttc = round(V.trajectory[ii, II_TTC], num_dp)
                dtc = round(V.trajectory[ii, II_DTC], num_dp)
                if ttc == graph.INF:
                    ttc = EMPTY_VALUE_STR
                if dtc == graph.INF:
                    dtc = EMPTY_VALUE_STR

                # Plot this vehicle
                v_plot_2.append(
                    g.plot_rectangle(
                        X=V.get_rectangle(ii), color="red", plot_heading=True, axis=ax2
                    )
                )

                # Plot annotations
                annot_color = "indigo"
                annot_string = rf"id={id}"
                if V.is_collision:
                    annot_color = "orange"

                # Plot the IDs next to the vehicles
                annotation_plot_2.append(
                    ax2.annotate(
                        annot_string,
                        (x + dx, y + dx),
                        size=6.5,
                        fontweight="bold",
                        zorder=20,
                        color=annot_color,
                    )
                )

                # Set the axes
                if V.current_state.vehicle_id == CENTRE_V_ID:
                    ax2.set_xlim([x - SCREEN_WIDTH / 2, x + SCREEN_WIDTH / 2])
                    ax2.set_ylim([y - SCREEN_HEIGHT / 2, y + SCREEN_HEIGHT / 2])

                    # Get x,y coordinates of the current time and plot it in the
                    # top right side of the screen
                    T = round(((ii + 1) / len_of_sim_2) * T_MAX_2, 1)
                    ax2.set_title(f"T={T}s")

        # Progress bar
        progressbar_anim(len_of_sim, ii + 1, prefix=f"Saving: ")

    if ii == len_of_sim_1:
        # Animation as finished, check if there are any collisions and
        # highlight them
        for V in v_list_1:
            if V.is_collision:
                g.plot_rectangle(
                    X=V.get_rectangle(len_of_sim_1 - 1),
                    color="orange",
                    plot_heading=True,
                    axis=ax1,
                )

    if ii == len_of_sim_2:
        # Animation as finished, check if there are any collisions and
        # highlight them
        for V in v_list_2:
            if V.is_collision:
                g.plot_rectangle(
                    X=V.get_rectangle(len_of_sim_2 - 1),
                    color="orange",
                    plot_heading=True,
                    axis=ax2,
                )


def main():
    # Global variabls
    global PLG_, v_list, len_of_sim, v_plot, annotation_plot

    # Plot PLG
    plt.sca(ax1)
    graph.draw(PLG_)
    plt.sca(ax2)
    graph.draw(PLG_)

    # Length of the simulation
    print(date_time.get_current_time(), "Saving animation")
    num_freeze_frames = int(FREEZE_FOR_X_SECONDS * FPS)
    anim = FuncAnimation(
        fig, animate, frames=len_of_sim + num_freeze_frames, interval=0
    )

    # Save the animation
    anim.save(TEST_SIM_SAVE_LOC + SIM_ANIM_NAME + ".gif", writer="pillow", fps=FPS)

    # TODO: Sometimes this script fails. Will fix...


if __name__ == "__main__":
    main()
