import sys
import os

# On my machine I need this line otherwise I get a "ModuleNotFoundError" when
# trying to import the other modules I have written within this directory.
sys.path.append(os.getcwd())

import functions.general as g
import classes.data as d
import math
import functions.date_time as date_time
from inputs import *
from fnames import *


###############################################################################
# Clean the dataset. If there are any two data points d_k and d_{k+1} where   #
# the Euclidean distance between them is greater than a specified threshold   #
# we will remove d_{k+1}. There are some anomalous data points like this in   #
# the datasets. These anomalous data points results in unrealistically larg   #
# edges in the PLG so we remove them using this data cleaning script.         #
#                                                                             #
# This function will take the original dataset loaded via the load_data class #
# and will return a cleaned dataset.                                          #
###############################################################################
def clean_data(original_dataset):
    # Define constants used for this function
    cleaned_dataset = d.data()
    dr_upper_threshold = 10
    num_vehicles = len((original_dataset.vehicle_sese)[:,0])
    new_vehicle_id = -1

    # Look through every vehicle and every vehicle's path and inspect change in
    # distance between adjacent time steps to remove anomalous data points.
    for ii_path in range(num_vehicles):
        for ii_sub_path in range((original_dataset.vehicle_sese)[ii_path, 1]):
            # We will re-name the vehicle IDs such that there is a single
            # vehicle ID corresponding to a single "path". Right now there are
            # multiple "paths" for each vehicle. In real life this corresponds
            # to the vehicle entering the map, exiting at a junction and then
            # re-entering the map at another junction. Its re-entry junction
            # usually corresponds to the one which it exited at, i.e., I 
            # for a vehicle to be able to get from it's original entry to its
            # final target it had to leave and re-enter the map. We will
            # separate all of these cases out and assign them their own vehicle
            # ID to make the coding a bit easier.
            new_vehicle_id += 1

            # Extract the data corresponding to each vehicle path
            x = g.se_extraction(original_dataset.vehicle_sese[ii_path, 0], original_dataset.x, original_dataset.vehicle_sese, sub_index=ii_sub_path)
            y = g.se_extraction(original_dataset.vehicle_sese[ii_path, 0], original_dataset.y, original_dataset.vehicle_sese, sub_index=ii_sub_path)
            lane_id = g.se_extraction(original_dataset.vehicle_sese[ii_path, 0], original_dataset.lane_id, original_dataset.vehicle_sese, sub_index=ii_sub_path)
            speed = g.se_extraction(original_dataset.vehicle_sese[ii_path, 0], original_dataset.speed, original_dataset.vehicle_sese, sub_index=ii_sub_path)
            acc = g.se_extraction(original_dataset.vehicle_sese[ii_path, 0], original_dataset.acc, original_dataset.vehicle_sese, sub_index=ii_sub_path)
            time = g.se_extraction(original_dataset.vehicle_sese[ii_path, 0], original_dataset.time, original_dataset.vehicle_sese, sub_index=ii_sub_path)

            # Intantiate variables to hold the new cleaned version of the data
            # for this path
            x_clean = [x[0,0]]
            y_clean = [y[0,0]]
            lane_id_clean = [lane_id[0,0]]
            speed_clean = [speed[0,0]]
            acc_clean = [acc[0,0]]
            time_clean = [time[0,0]]
            vehicle_id_clean = [new_vehicle_id]

            # Number of data points in this vehicle path
            path_length = len(x)

            # Cycle through the data for this path
            for ii in range(1, path_length):
                # Current x,y values
                x_current = x[ii,0]
                y_current = y[ii,0]
                # Previous x,y values
                x_prev = x_clean[-1]
                y_prev = y_clean[-1]

                # Euclidean distance between current and previous datum
                dr = math.sqrt((x_current - x_prev)**2 + (y_current - y_prev)**2)

                # If the distance between the current data point and the
                # previous data point is less than the threshold then append
                # this data point to the "cleaned" versions. Otherwise ignore
                # it.
                if dr < dr_upper_threshold:
                    x_clean.extend(x[ii])
                    y_clean.extend(y[ii])
                    lane_id_clean.extend(lane_id[ii])
                    speed_clean.extend(speed[ii])
                    acc_clean.extend(acc[ii])
                    time_clean.extend(time[ii])
                    vehicle_id_clean.append(new_vehicle_id)

            # Append the cleaned vehicle data to our data object
            (cleaned_dataset.x).extend(x_clean)
            (cleaned_dataset.y).extend(y_clean)
            (cleaned_dataset.lane_id).extend(lane_id_clean)
            (cleaned_dataset.speed).extend(speed_clean)
            (cleaned_dataset.acc).extend(acc_clean)
            (cleaned_dataset.time).extend(time_clean)
            (cleaned_dataset.vehicle_id).extend(vehicle_id_clean)

    # Build the sese matrices for this new cleaned dataset
    cleaned_dataset.vehicle_sese = g.get_se_matrix(cleaned_dataset.vehicle_id)
    cleaned_dataset.lane_sese = g.get_se_matrix(cleaned_dataset.lane_id)

    # Set num_data_points
    cleaned_dataset.num_data_points = len(cleaned_dataset.x)

    return cleaned_dataset


def main():
    # Time the script
    print(date_time.get_current_time(), "Program started")

    # Load the original dataset
    original_dataset = d.load_data()
    print(date_time.get_current_time(), "Loaded original dataset")

    # Clean the dataset
    cleaned_dataset = clean_data(original_dataset)
    print(date_time.get_current_time(), "Finished cleaning data")

    # Save data
    g.save_pickled_data(CLEAN_DATA_LOC+CLEAN_DATA_NAME, cleaned_dataset)
    print(date_time.get_current_time(), "Saved clean data")


if __name__=="__main__":
    main()


