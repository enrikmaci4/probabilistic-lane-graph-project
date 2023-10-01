import sys
import os

# On my machine I need this line otherwise I get a "ModuleNotFoundError" when
# trying to import the other modules I have written within this directory.
sys.path.append(os.getcwd())

import functions.general as g
from functions.general import progressbar_anim
import functions.date_time as date_time
import functions.graph as graph
import functions.simulation as sim
from animation.animation import EMPTY_VALUE_STR, NEWLINE_CHAR
from matplotlib.animation import FuncAnimation
import time
import matplotlib.pyplot as plt
import random
from inputs import *
from fnames import *
import numpy as np
from classes.PLG import *
from classes.vehicle import *
import models.acceleration as acc_models
from animation.animation import animate


###############################################################################
# ABOUT THIS SCRIPT:                                                          #
#                                                                             #
# - Generate and save many simulations and save them according to whether     #
#   they terminated in a collision or not.                                    #
# - The simulations are generated from a random initial state.                #
# - The output from this script is saved into output/set1.                    #
# - The initial state data is stored in the files ending in "is".             #
# - We will then re-run simulations with the exact same initial state under a #
#   different model and save the output of this model in set2.                #
# - The script test_simulation_3.py is what we will use to take the outputs   #
#   of this script, run another simulation under a different model, and save  #
#   it to set 2.                                                              #
#                                                                             #
###############################################################################

###############################################################################
# Animation functions.                                                        #
###############################################################################
# Recommended FPS for smooth animations: 10
FPS = 10
FREEZE_FOR_X_SECONDS = 3
dx = 0.75
dy = 0.1
CENTRE_V_ID = 0
# GLOBAL VARIABLES NEEDED FOR ANIMATION
# - PLG object
PLG_ = None
# - Load vehicle data
v_list = None
len_of_sim = None
# - Store the plotted vehicles so we can delete them before we plot them again
v_plot = None
# - Annotation
annotation_plot = None
# - Time plot
time_plot = [None]
# - File locations to load/save data to
SAVE_LOC = None
SAVE_SUFF = None
# - Counter variables
II = None
II_START = 30
II_COUNTER = 0
num_cc = None
num_ncc = None


# Initializing a figure in which the graph will be plotted
fig = plt.figure()
plt.gca().set_aspect("equal", adjustable="box")


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

            # Delete plot
            v_plot_[0].remove()
            annotation_plot_.remove()

            # Pop from list
            v_plot.pop(jj)
            annotation_plot.pop(jj)

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
            annot_string = rf"ttc={ttc}{NEWLINE_CHAR}dtc={dtc}{NEWLINE_CHAR}v={speed}{NEWLINE_CHAR}id={id}{NEWLINE_CHAR}"
            annotation_plot.append(
                plt.annotate(
                    annot_string,
                    (x + dx, y + dx),
                    size=6.5,
                    fontweight="bold",
                    zorder=20,
                    color="indigo",
                )
            )

            # Set the axes
            if V.current_state.vehicle_id == CENTRE_V_ID:
                plt.xlim([x - SCREEN_WIDTH / 2, x + SCREEN_WIDTH / 2])
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
        progressbar_anim(len_of_sim, ii + 1, prefix=f"Saving: {II} ")

    elif ii == len_of_sim:
        # Animation as finished, check if there are any collisions and
        # highlight them
        for V in v_list:
            if V.is_collision:
                g.plot_rectangle(
                    X=V.get_rectangle(len_of_sim - 1), color="orange", plot_heading=True
                )


def save_animation():
    # Global variabls
    global PLG_, v_list, len_of_sim

    # Plot PLG
    graph.draw(PLG_)

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
    anim.save(f"{SAVE_LOC}anim_{II}{SAVE_SUFF}.gif", writer="pillow", fps=FPS)


def main():
    # Initialisations
    global II, PLG_, START_ANIMATION, v_list, len_of_sim, v_plot, annotation_plot, time_plot, SAVE_LOC, SAVE_SUFF, num_cc, num_ncc

    # GLOBAL VARIABLES NEEDED FOR ANIMATION
    # - PLG object
    PLG_ = g.load_pickled_data(PLG_SAVE_LOC + PLG_NAME)
    # - Load vehicle data
    v_list = g.load_pickled_data(TEST_SIM_SAVE_LOC + SIM_DATA_PKL_NAME)
    len_of_sim = v_list[0].trajectory_length
    # - Store the plotted vehicles so we can delete them before we plot them
    #   again
    v_plot = []
    # - Annotation
    annotation_plot = []
    # - File locations to load/save data to
    SAVE_LOC = SET1_SAVE_LOC
    SAVE_SUFF = ""
    # - Counter variables
    II = II_START
    II_COUNTER = 0
    num_cc = 0
    num_ncc = 0

    # Use lists to store the corner cases/non corner cases
    cc_list = []
    ncc_list = []

    # Create a PLG object
    PLG_ = g.load_pickled_data(PLG_SAVE_LOC + PLG_NAME)
    print(date_time.get_current_time(), "Loaded PLG")

    # Generate and save simulations
    while II_COUNTER < NUM_SIMULATIONS:
        # Try, if fail try again with another
        try:
            # Generate the simulation
            print()
            is_cc = sim.generate_single_simulation(
                PLG_, SAVE_LOC=SAVE_LOC, II=str(II), generate_only_cc=False
            )
            if is_cc:
                SAVE_SUFF = CC_SUFF
                cc_list.append(II)
                num_cc += 1
            else:
                SAVE_SUFF = NCC_SUFF
                ncc_list.append(II)
                num_ncc += 1

            # Re-assign global variables
            load_loc = f"{SAVE_LOC}{SIM_DATA_PKL_NAME}_{II}{SAVE_SUFF}"
            v_list = g.load_pickled_data(load_loc)
            len_of_sim = v_list[0].trajectory_length
            v_plot = []
            annotation_plot = []
            START_ANIMATION = False

            # Save the animation
            plt.cla()
            save_animation()
            II += 1
            II_COUNTER += 1

        # We're tired of this - break!
        except KeyboardInterrupt:
            break

        # Woops. There was an assert we were too lazy to handle. Try again :)
        except Exception:
            pass

        # Write stats to text file every iteration so we can get live stats
        with open(f"{SAVE_LOC}{STATS_NAME}", "w+") as stats_file:
            # First write the corner case stats
            stats_file.write(f"Num CCs: {num_cc} out of Total: {num_cc+num_ncc}\n\n")

            # Now write the corner cases:
            stats_file.write(f"Corner cases:\n")
            for cc in cc_list:
                stats_file.write(f"{cc}\n")
            stats_file.write(f"\n")

            # Now write the no corner cases:
            stats_file.write(f"No corner cases:\n")
            for ncc in ncc_list:
                stats_file.write(f"{ncc}\n")
            stats_file.write(f"\n")


if __name__ == "__main__":
    main()
