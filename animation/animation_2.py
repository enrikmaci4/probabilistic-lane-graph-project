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


NEWLINE_CHAR = "\n"
EMPTY_VALUE_STR = "---"


###############################################################################
# Generate an animation from an output dataset.                               #
#                                                                             #
# This script loads the scenario in TEST_SIM_SAVE_LOC, create sa GIF and      #
# saves it to the same output directory.                                      #
#                                                                             #
# This is similar to animation.py except it can load an animation from any    #
# location, i.e., TEST_SIM_SAVE_LOC or SETi_SAVE_LOC.                         #
#                                                                             #
###############################################################################


# INPUT: Define which simulation to plot using
#
#   SET = TEST_SIM_SAVE_LOC or SETi_SAVE_LOC
#   II = None or integer. None for TEST folder and integer for SET folder
#
SET = TEST_SIM_SAVE_LOC
II = None

if II == None:
    LOAD_DATA_LOC = f"{TEST_SIM_SAVE_LOC}{SIM_DATA_PKL_NAME}"
else:
    LOAD_DATA_LOC = f"{SET}{SIM_DATA_PKL_NAME}_{II}"


###############################################################################
# Generate an animation from an output dataset.                               #
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
try:
    v_list = g.load_pickled_data(f"{LOAD_DATA_LOC}{NCC_SUFF}")
except FileNotFoundError:
    try:
        v_list = g.load_pickled_data(f"{LOAD_DATA_LOC}{CC_SUFF}")
    except FileNotFoundError:
        v_list = g.load_pickled_data(f"{LOAD_DATA_LOC}")
len_of_sim = v_list[0].trajectory_length
# - Store the plotted vehicles so we can delete them before we plot them
#   again
v_plot = []
# - Annotation
annotation_plot = []
annotation_plot_2 = []
# - Time plot
time_plot = [None]


# Animation function
def animate(ii):
    # Global vars
    global START_ANIMATION, PLG_, v_list, len_of_sim

    if not START_ANIMATION:
        START_ANIMATION = True

    elif ii < len_of_sim:
        # Before plotting, remove the previous plots
        num_plots = len(v_plot)
        for nn in range(num_plots):
            # Start indexing from the final element because we're going to
            # remove
            jj = num_plots - 1 - nn

            # Get the matplotlib objects
            v_plot_ = v_plot[jj]
            annotation_plot_ = annotation_plot[jj]
            annotation_plot_2_ = annotation_plot_2[jj]

            # Delete plot
            v_plot_[0].remove()
            annotation_plot_.remove()
            annotation_plot_2_.remove()

            # Pop from list
            v_plot.pop(jj)
            annotation_plot.pop(jj)
            annotation_plot_2.pop(jj)

            # There's only 1 time plot so remove it on the final iteration
            if jj == 0:
                # Delete time in top right corner
                time_plot[0].remove()

        # Cycle through vehicle list and plot the vehicle
        for V in v_list:
            num_dp = 1
            # Get some constants
            id = int(V.trajectory[ii, II_VEHICLE_ID])
            x = V.trajectory[ii, II_X]
            y = V.trajectory[ii, II_Y]
            speed = round(V.trajectory[ii, II_SPEED], num_dp)
            acc = round(V.trajectory[ii, II_ACC], num_dp)
            ttc = round(V.trajectory[ii, II_TTC], num_dp)
            dtc = round(V.trajectory[ii, II_DTC], num_dp)
            head_ang = round(V.trajectory[ii, II_HEAD_ANG] * 180 / math.pi, num_dp)
            if ttc == graph.INF:
                ttc = EMPTY_VALUE_STR
            if dtc == graph.INF:
                dtc = EMPTY_VALUE_STR

            # Plot this vehicle
            v_plot.append(
                g.plot_rectangle(X=V.get_rectangle(ii), color="red", plot_heading=True)
            )

            # Plot annotations
            annot_color = "indigo"
            annot_string_id = rf"id={id}"
            annot_string = rf"| id={id} | ttc={ttc} | dtc={dtc} | v={speed} | a={acc} |"
            if V.is_collision:
                annot_color = "orange"

            # Plot the IDs next to the vehicles
            annotation_plot.append(
                plt.annotate(
                    annot_string_id,
                    (x + dx, y + dx),
                    size=6.5,
                    fontweight="bold",
                    zorder=20,
                    color=annot_color,
                )
            )
            # Plot the vehicle kinematics on the right
            annotation_plot_2.append(
                plt.annotate(
                    annot_string,
                    (
                        v_list[0].trajectory[ii, II_X] + 0.7 * SCREEN_WIDTH,
                        v_list[0].trajectory[ii, II_Y]
                        + SCREEN_HEIGHT / 2
                        - 2
                        - 2 * V.current_state.vehicle_id,
                    ),
                    size=6.5,
                    fontweight="bold",
                    zorder=20,
                    color=annot_color,
                )
            )

            # Set the axes
            if V.current_state.vehicle_id == CENTRE_V_ID:
                plt.xlim([x - SCREEN_WIDTH / 2, x + 3 * SCREEN_WIDTH / 2])
                plt.ylim([y - SCREEN_HEIGHT / 2, y + SCREEN_HEIGHT / 2])

                # Get x,y coordinates of the current time and plot it in the
                # top right side of the screen
                x_time = x + SCREEN_WIDTH / 2
                y_time = y + SCREEN_HEIGHT / 2
                T = round(((ii + 1) / len_of_sim) * SIM_LENGTH, 1)
                time_plot[0] = plt.annotate(
                    f"T={T}s", (x_time, y_time), size=10, color="black"
                )

        # Progress bar
        progressbar_anim(len_of_sim, ii + 1, prefix=f"Saving: ")

    elif ii == len_of_sim:
        # Animation as finished, check if there are any collisions and
        # highlight them
        for V in v_list:
            if V.is_collision:
                g.plot_rectangle(
                    X=V.get_rectangle(len_of_sim - 1), color="orange", plot_heading=True
                )


# Initializing a figure in which the graph will be plotted
fig = plt.figure()
plt.gca().set_aspect("equal", adjustable="box")


def main():
    # Global variabls
    global PLG_, v_list, len_of_sim, v_plot, annotation_plot

    # Plot PLG
    graph.draw(PLG_)

    # If this is a special side case where a vehicle is running a red light,
    # set to True. This variable is a special edge case I've used to generate
    # artificial simulations and shouldn't be set to True for any mainline
    # cases.
    red_light = False
    # Red light and green light
    red_light = [37, 38, 39, 40, 41, 42]
    green_light = [1050, 31, 299, 512, 484]
    if red_light:
        plt.scatter(
            PLG_.nodes[red_light, 0],
            PLG_.nodes[red_light, 1],
            s=20,
            color="red",
            zorder=40,
        )
        plt.scatter(
            PLG_.nodes[green_light, 0],
            PLG_.nodes[green_light, 1],
            s=20,
            color="green",
            zorder=40,
        )

    # Length of the simulation
    print(date_time.get_current_time(), "Saving animation")
    num_freeze_frames = int(FREEZE_FOR_X_SECONDS * FPS)
    anim = FuncAnimation(
        fig, animate, frames=len_of_sim + num_freeze_frames, interval=0
    )

    # Define some plot params
    # - Hide X and Y axes tick marks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    # - Reduce the whitespace around the plot
    plt.tight_layout(h_pad=0.1, w_pad=0.1)

    # Save the animation
    anim.save(TEST_SIM_SAVE_LOC + SIM_ANIM_NAME + ".gif", writer="pillow", fps=FPS)

    # TODO: Sometimes this script fails. Will fix...


if __name__ == "__main__":
    main()
