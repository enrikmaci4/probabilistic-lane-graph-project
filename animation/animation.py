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
###############################################################################
# ANIMATION CONSTANTS
START_ANIMATION = False
# Recommended FPS for smooth animations: 10
FPS = 10
FREEZE_FOR_X_SECONDS = 3
dx = 0.75
dy = 0.1
# GLOBAL VARIABLES NEEDED FOR ANIMATION
# - PLG object
PLG_ = g.load_pickled_data(PLG_SAVE_LOC+PLG_NAME)
# - Load vehicle data
v_list = g.load_pickled_data(TEST_SIM_SAVE_LOC+SIM_DATA_PKL_NAME)
len_of_sim = v_list[0].trajectory_length
# - Store the plotted vehicles so we can delete them before we plot them again
v_plot = []
# - Annotation
annotation_plot = []


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
            jj = num_plots-1-nn

            # Get the matplotlib objects
            v_plot_ = v_plot[jj]
            annotation_plot_ = annotation_plot[jj]

            # Delete plot
            v_plot_[0].remove()
            annotation_plot_.remove()

            # Pop from list
            v_plot.pop(jj)
            annotation_plot.pop(jj)

        # Cycle through vehicle list and plot the vehicle
        for V in v_list:
            num_dp = 3
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
            v_plot.append(g.plot_rectangle(X=V.get_rectangle(ii), color="red", plot_heading=True))

            # Plot annotations
            annot_string = rf"ttc={ttc} | dtc={dtc}{NEWLINE_CHAR}v={speed} | id={id}"
            annotation_plot.append(plt.annotate(annot_string, (x+dx, y+dx), size=6.5, fontweight="bold", zorder=20, color="indigo"))
    
            # Set the axes
            if V.current_state.vehicle_id == 0:
                plt.xlim([x-SCREEN_WIDTH/2, x+SCREEN_WIDTH/2])
                plt.ylim([y-SCREEN_HEIGHT/2, y+SCREEN_HEIGHT/2])

        # Progress bar
        progressbar_anim(len_of_sim, ii+1, prefix="Saving: ")
        
    elif ii == len_of_sim:
        # Animation as finished, check if there are any collisions and
        # highlight them
        for V in v_list:
            if V.is_collision:
                g.plot_rectangle(X=V.get_rectangle(len_of_sim-1), color="orange", plot_heading=True)
  

# Initializing a figure in which the graph will be plotted
fig = plt.figure() 
plt.gca().set_aspect("equal", adjustable="box")


def main():
    # Global variabls
    global PLG_, v_list, len_of_sim

    # Plot PLG
    graph.draw(PLG_)

    # Length of the simulation
    print(date_time.get_current_time(), "Saving animation")
    num_freeze_frames = int(FREEZE_FOR_X_SECONDS*FPS)
    anim = FuncAnimation(fig, animate, frames=len_of_sim+num_freeze_frames, interval=0)

    # Define some plot params
    # - Hide X and Y axes tick marks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    # - Reduce the whitespace around the plot
    plt.tight_layout(h_pad=0.1, w_pad=0.1)

    # Save the animation
    anim.save(TEST_SIM_SAVE_LOC+SIM_ANIM_NAME+".gif", writer='pillow', fps=FPS)

    # TODO: Sometimes this script fails. Will fix...


if __name__=="__main__":
    main()
