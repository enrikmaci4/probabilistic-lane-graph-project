import numpy as np
from sklearn.cluster import KMeans
from inputs import *


###############################################################################
# cluster_generation:                                                         #
#                                                                             #
# Purpose: Generate a dictionary of {cluster id: [list of nodes in cluster]}  #
#          for both the start and target clusters. Start and target clusters  #
#          are used to determine to entry and exit points of the map as nodes #
#          in the PLG.                                                        #
#                                                                             #
# Params: IN/OUT PLG  - A PLG object of type "PLG" defined in classes/PLG.py. #
#                       The PLG.start_cluster and PLG.target_cluster          #
#                       parameterts will be updated with the start and target #
#                       clusters generated by this function.                  #  
#                                                                             # 
###############################################################################
def cluster_generation(PLG):
    start_nodes = []
    target_nodes = []
    start_clusters = {ii:[] for ii in range(NUM_START_CLUSTERS)}
    target_clusters = {ii:[] for ii in range(NUM_TARGET_CLUSTERS)}
    closest_clusters_dict = {ii:[] for ii in range(NUM_TARGET_CLUSTERS)}
    num_kmeans_iterations = 1000
    
    # First get a list of all starting and target nodes
    for path_ii in PLG.vehicle_paths:
        start_nodes.append(PLG.vehicle_paths[path_ii][0])
        target_nodes.append(PLG.vehicle_paths[path_ii][-1])

    # We will remove repeated nodes from the start and target nodes so thaw we
    # have a chance of detecting the less frequency entry/exit points in the
    # map. Otherwise, if there is one entry/exit with a lot of traffic going
    # it may dominate the clustering and take two cluster centres as opposed to
    # one.
    start_nodes = np.unique(start_nodes, axis=0)
    target_nodes = np.unique(target_nodes, axis=0)

    # Get the coordinates of the start and target nodes
    start_node_coords = PLG.nodes[start_nodes]
    target_node_coords = PLG.nodes[target_nodes]

    # We now do a k-means clustering on the start and target nodes
    # Start nodes
    kmeans_start = KMeans(n_clusters=NUM_START_CLUSTERS, init='k-means++', n_init=1, max_iter=num_kmeans_iterations).fit(start_node_coords)
    start_node_cluster_labels = kmeans_start.labels_

    # Target nodes
    kmeans_target = KMeans(n_clusters=NUM_TARGET_CLUSTERS, init='k-means++', n_init=1, max_iter=num_kmeans_iterations).fit(target_node_coords)
    target_node_cluster_labels = kmeans_target.labels_

    # Build the start and target cluster dictionaries
    # Start nodes
    num_start_nodes = len(start_nodes)
    for ii in range(num_start_nodes):
        start_clusters[start_node_cluster_labels[ii]].append(start_nodes[ii])

    # Target nodes
    num_target_nodes = len(target_nodes)
    for ii in range(num_target_nodes):
        target_clusters[target_node_cluster_labels[ii]].append(target_nodes[ii])

    # Finally, we need to generate a list of closest clusters for each cluster
    for ii in closest_clusters_dict:
        # Create a copy of the cluster centres
        cluster_centre_coords = kmeans_target.cluster_centers_

        # Coordinates of the current cluster centre
        centre_ii = cluster_centre_coords[ii,:]

        # Calculate which cluster centre is closest to the current cluster
        # centre
        dist_to_centre_ii = np.linalg.norm(cluster_centre_coords - centre_ii, ord=2, axis=1)

        # Get a list of indices from 0 to NUM_TARGET_CLUSTERS-1 which we will
        # use to sort the distances
        closest_cluster_list = np.arange(NUM_TARGET_CLUSTERS)

        # Now sort the indices based on the distances
        closest_cluster_list = closest_cluster_list[np.argsort(dist_to_centre_ii)]

        # Assign the closest clusters to the dictionary
        closest_clusters_dict[ii] = closest_cluster_list

    # Assign the start and target clusters to the PLG object
    PLG.start_cluster_centres = kmeans_start.cluster_centers_
    PLG.target_cluster_centres = kmeans_target.cluster_centers_
    PLG.start_clusters = start_clusters
    PLG.target_clusters = target_clusters
    PLG.closest_clusters_dict = closest_clusters_dict

    return True

