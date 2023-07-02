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
import random
from inputs import *
import numpy as np
from classes.PLG import *
from classes.vehicle import *


DATA_LOC = "data/"+DATASET+"/cleaned/"
PLG_SAVE_LOC = "data/"+DATASET+"/data-structures/"


###############################################################################
# Use this function to generate the initial vehicle position. This will be    #
# our "AV". However, all vehicles will run under the same policy in this test #
# so the term "AV" is really just synonymous with the "subject vehicle", i.e. #
# the one that we will focus on when we generate our visualisations.          #
#                                                                             #
# Returns:                                                                    #
#                                                                             #
# av - A "Vehicle" type object with an initial state.                         #
#                                                                             #
###############################################################################
def initialise_av_position(PLG_: PLG) -> Vehicle:
    # First we're going to generate a single vehicle path from an entry point
    # to an exit. We will then choose a random node, somewhere in the middle
    # of the path, and choose this as the initial state of the AV. There are a
    # few reasons for this:
    # - The entry and exit locations can be a bit funky and we'd rather
    #   generate cleaner simulations for now, atleast until we understand the
    #   model better.
    # - We don't really want vehicle's to suddenly disappear from the map which
    #   is what will happen if a vehicle reaches it's target destination.
    start_cluster = int(np.random.choice(list(PLG_.start_clusters.keys())))
    start_node = int(np.random.choice(PLG_.start_clusters[start_cluster]))
    # We want kind of long paths to get a nice long simulation so choose 
    # one of the target clusters which is further away.
    start_coords = PLG_.nodes[start_node,:]
    distance_from_start_to_targets = np.sum(np.square(PLG_.target_cluster_centres - start_coords), axis=1)
    # TODO: The way we select this could probable improve to get more
    # variation in target clusters. E.g. randomly choose one betwee the mid-
    # point and the end in a list ordered from lowest to highest (or vice
    # versa).
    target_cluster = np.argmax(distance_from_start_to_targets)

    # Generate a path
    path = graph.path_generation(PLG, start_node, target_cluster)

    # Now choose a random node between the a_low and a_upp percentiles. This
    # is because we want to initialise the AV somewhere in the middle of it's
    # path.
    a_low = 0.25
    a_upp = 0.75
    a_low_ind = int(a_low*len(path))
    a_upp_ind = int(a_upp*len(path))
    initial_node = np.random.choice(path[a_low_ind:a_upp_ind])
    initial_node_index = path.index(initial_node)
    path_trunc = graph.path_generation(PLG, initial_node, target_cluster, max_path_length=15)

    # Get an "output_data" data structure. This data structure converts a node
    # path into a 2D matrix with columns [x, y, heading angle]. We're mainly
    # interested in the heading angle.
    output_data = graph.node_path_to_output_data(PLG, path)

    # Create a DataRow data structure and populate it with the data we need to
    # initialise the AV.
    initial_state = DataRow()
    initial_state.vehicle_id = 0
    initial_state.time = 0
    initial_state.x = output_data[initial_node_index, 0]
    initial_state.y = output_data[initial_node_index, 1]
    initial_state.node = initial_node
    initial_state.lane_id = PLG_.node_lane_ids[initial_node]


    graph.draw(PLG)
    graph.plot_node_path(PLG, path)
    graph.plot_node_path(PLG, path_trunc, color="skyblue")
    plt.show()
    pass


def main():
    # Time the script
    t_start = time.time()
    print(date_time.get_current_time(), "Program started")

    # Load the cleaned data
    data = g.load_pickled_data(DATA_LOC+"clean_data")
    print(date_time.get_current_time(), "Loaded clean data")

    # Create a PLG object
    PLG = g.load_pickled_data(PLG_SAVE_LOC+"PLG")
    print(date_time.get_current_time(), "Loaded PLG")

    # Print time take
    print(f"{date_time.get_current_time()} Time taken to load data = {round(time.time() - t_start, 3)} s")

    # Initialise an AV
    AV = initialise_av_position(PLG)

    # Set the aspect ratio to be equal
    #plt.gca().set_aspect("equal", adjustable="box")
    #plt.show()


if __name__=="__main__":
    main()

