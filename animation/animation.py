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
import numpy as np
from classes.PLG import *
from classes.vehicle import *


DATA_LOC = "data/"+DATASET+"/cleaned/"
PLG_SAVE_LOC = "data/"+DATASET+"/data-structures/"
SIM_DATA_SAVE_LOC = "output-data/simulation/"


###############################################################################
# Generate an animation from an output dataset.                               #
###############################################################################
# ANIMATION CONSTANTS
START_ANIMATION = False
dx = 0.75
dy = 0.1
# GLOBAL VARIABLES NEEDED FOR ANIMATION
# - PLG object
PLG_ = g.load_pickled_data(PLG_SAVE_LOC+"PLG")
# - Load vehicle data
data = np.loadtxt(SIM_DATA_SAVE_LOC+"test_data")
v_list = g.load_pickled_data(SIM_DATA_SAVE_LOC+"test_list")
# - Store the plotted vehicles so we can delete them before we plot them again
v_plot = []
# - Annotation
annotation_plot = []


# Animation function
def animate(ii):
    global START_ANIMATION, PLG_, v_list

    if not START_ANIMATION:
        START_ANIMATION = True
    else:
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
            # Get some constants
            id = int(V.trajectory[ii, II_VEHICLE_ID])
            x = V.trajectory[ii, II_X]
            y = V.trajectory[ii, II_Y]
            speed = round(V.trajectory[ii, II_SPEED], 3)
            acc = round(V.trajectory[ii, II_ACC], 3)
            ttc = round(V.trajectory[ii, II_TTC], 3)
            if ttc == graph.INF_TTC:
                ttc = "---"

            # Plot this vehicle
            v_plot.append(g.plot_rectangle(X=V.get_rectangle(ii), color="red", plot_heading=True))

            # Plot annotations
            annot_string = f"ttc={ttc}\nv={speed}\nid={id}"
            annotation_plot.append(plt.annotate(annot_string, (x+dx, y+dx), size=6.5, fontweight="bold", zorder=20, color="indigo"))
    
            # Set the axes
            if V.current_state.vehicle_id == 0:
                plt.xlim([x-SCREEN_WIDTH/2, x+SCREEN_WIDTH/2])
                plt.ylim([y-SCREEN_HEIGHT/2, y+SCREEN_HEIGHT/2])
        
        # Print the index
        if (ii+1)%10 == 0:
            print(ii+1)
        

# Initializing a figure in which the graph will be plotted
fig = plt.figure() 
plt.gca().set_aspect("equal", adjustable="box")


def main():
    # Global variabls
    global PLG_, v_list

    # Plot PLG
    graph.draw(PLG_)

    # Length of the simulation
    len_of_sim = v_list[0].trajectory_length

    anim = FuncAnimation(fig, animate, frames=len_of_sim, interval=0)
    
    # Define some plot params
    # - Hide X and Y axes tick marks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    # - Reduce the whitespace around the plot
    plt.tight_layout(h_pad=0.1, w_pad=0.1)

    # Save the animation
    anim.save(SIM_DATA_SAVE_LOC+'test.gif', writer='pillow', fps=1)

    # TODO: Sometimes this script fails. Will fix...

    # Plot one of the vehicles trajectories
    #V = v_list[0]
    #plt.scatter(V.trajectory[:,II_X], V.trajectory[:,II_Y], s=10, color="skyblue", zorder=20)
    #plt.xlim([V.current_state.x-SCREEN_WIDTH/2, V.current_state.x+SCREEN_WIDTH/2])
    #plt.ylim([V.current_state.y-SCREEN_HEIGHT/2, V.current_state.y+SCREEN_HEIGHT/2])
    #plt.show()


if __name__=="__main__":
    main()
