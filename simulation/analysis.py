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
from sklearn.decomposition import PCA


###############################################################################
# ABOUT THIS SCRIPT:                                                          #
#                                                                             #
# - Performs a numerical analysis on the CC simulation output data.           #
#                                                                             #
###############################################################################


# File to load simulation data from
LOAD_LOC = SET1_SAVE_LOC


def calculate_rr_lon(V1: Vehicle, V2:Vehicle, ii: int):
    """Calulcate longitudinal relative distance between V1 and V2 for the ii'th
    time step.
    """
    # Heading angle of first vehicle
    h1 = V1.trajectory[ii, II_HEAD_ANG]
    # X,Y coords of the two vehicles
    x1 = V1.trajectory[ii, II_X]
    y1 = V1.trajectory[ii, II_Y]
    x2 = V2.trajectory[ii, II_X]
    y2 = V2.trajectory[ii, II_Y]
    # Distance between the two vehicles as a vector
    d_vec = np.array([x1 - x2, y1 - y2])
    # Unit vector in the direction of h1
    u_vec = np.array([math.cos(h1), math.sin(h1)])
    # The longitudinal distance is the projection of d_vec onto u_vec. I.e.
    # the proportion of the absolute distance which is in the direction of h1
    return abs(np.dot(d_vec, u_vec))


def calculate_rr_lat(V1: Vehicle, V2:Vehicle, ii: int):
    """Calulcate lateral relative distance between V1 and V2 for the ii'th
    time step. The lateral velocity is calculated in the exact same way as the
    longitudinal velocity, however, we rotate the unit vector by pi/2 radians
    to get a perpendicular vector.
    """
    # Heading angle of first vehicle
    h1 = V1.trajectory[ii, II_HEAD_ANG]
    # X,Y coords of the two vehicles
    x1 = V1.trajectory[ii, II_X]
    y1 = V1.trajectory[ii, II_Y]
    x2 = V2.trajectory[ii, II_X]
    y2 = V2.trajectory[ii, II_Y]
    # Distance between the two vehicles as a vector
    d_vec = np.array([x1 - x2, y1 - y2])
    # Unit vector in the direction of h1
    u_vec = np.array([math.cos(h1 + math.pi/2), math.sin(h1 + math.pi/2)])
    # The lateral distance is the projection of d_vec onto u_vec. I.e.
    # the proportion of the absolute distance which is in the direction of
    # h1 + pi/2
    return abs(np.dot(d_vec, u_vec))


def main():
    # Initialisations
    PLG_ = g.load_pickled_data(PLG_SAVE_LOC+PLG_NAME)
    # - A feature vector for a single timestep will have:
    #   [rr_lon, rr_lat, rr, v_rel, h]
    num_features_per_ii = 5
    # - We extract the top feature vector at multiple time steps to get the
    #   entire feature vector for the whole PCA method. We'll do this for 1
    #   second before the crash.
    T_feature_extraction = 1
    num_time_steps = math.floor(T_feature_extraction/dt)
    num_features = num_features_per_ii*num_time_steps
    pca_feature_mat = np.zeros((0, num_features))
    # - Create a PLG object
    PLG_ = g.load_pickled_data(PLG_SAVE_LOC+PLG_NAME)
    II = 0
    print(date_time.get_current_time(), "Loaded PLG")

    # Load simulation data and build the PCA matrix
    while II < NUM_SIMULATIONS:
        # Keyboard interrupt was interrupting as it should
        try:
            # Load vehicle file
            v_list = g.load_pickled_data(f"{LOAD_LOC}{SIM_DATA_PKL_NAME}_{II}{CC_SUFF}")

            # Initialise a feature vector
            feature_vector_ii = np.zeros((1, num_features))
            
            # Find the two vehicles involved in the crash
            ii = 0
            num_v_in_cc_found = 0
            while ii < len(v_list):
                if num_v_in_cc_found == 0:
                    # Look for first vehicle
                    if v_list[ii].is_collision:
                        V1 = v_list[ii]
                        num_v_in_cc_found += 1

                elif num_v_in_cc_found == 1:
                    # Look for second vehicle
                    if v_list[ii].is_collision:
                        V2 = v_list[ii]
                        num_v_in_cc_found += 1
                        break

                # Increment counter
                ii += 1
            
            # Extract the feature vector for each time step
            for ii in range(V1.trajectory_length):
                # Start from the final time step
                jj = V1.trajectory_length - 1 - ii

                # Calculate the features
                rr_lon = calculate_rr_lon(V1, V2, jj)
                rr_lat = calculate_rr_lat(V1, V2, jj)
                rr_vel = abs(V1.trajectory[jj, II_SPEED] - V2.trajectory[jj, II_SPEED])
                h1 = V1.trajectory[jj, II_HEAD_ANG]
                h2 = V2.trajectory[jj, II_HEAD_ANG]

                # Now fill the feature vector
                ii_start = ii*num_features_per_ii
                feature_vector_ii[0, ii_start + 0] = rr_lon
                feature_vector_ii[0, ii_start + 1] = rr_lat
                feature_vector_ii[0, ii_start + 2] = rr_vel
                feature_vector_ii[0, ii_start + 3] = h1
                feature_vector_ii[0, ii_start + 4] = h2

                # Break if we've filled the feature vector
                if ii_start + num_features_per_ii == num_features:
                    break

            # Stack this row onto the PCA matrix
            pca_feature_mat = np.vstack((pca_feature_mat, feature_vector_ii))

        except KeyboardInterrupt:
            # Keyboard interrupt sent, quit
            quit()

        except FileNotFoundError:
            # File doesn't exist, increment II and try again
            pass

        # Increment II
        II += 1

    # Initialize PCA with the number of components you want to retain
    num_components = 2
    pca = PCA(n_components=num_components)
    
    # Fit the PCA model to the data
    pca.fit(pca_feature_mat)
    
    # Transform the original data to the lower-dimensional representation
    transformed_data = pca.transform(pca_feature_mat)
    
    # The principal components (eigenvectors) are available in pca.components_
    principal_components = pca.components_
    
    # The amount of variance explained by each component can be found in
    # pca.explained_variance_ratio_
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Print the transformed data and other results
    print("Original data:\n", pca_feature_mat)
    print("\nTransformed data:\n", transformed_data)
    print(f"\nPrincipal Components:\n{np.shape(principal_components)}\n", principal_components)
    print("\nExplained Variance Ratio:\n", explained_variance_ratio)

    plt.scatter(transformed_data[:,0], transformed_data[:,1], s=10, color="red")
    plt.show()


if __name__=="__main__":
    main()

