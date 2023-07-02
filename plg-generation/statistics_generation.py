import numpy as np
from classes.PLG import *
from classes.data import *


###############################################################################
# statistics_generation:                                                      #
#                                                                             #                             
# Purpose: Generate some simple statistics for this dataset.                  #    
#                                                                             #                                    
# Params: IN    data - A data object of type "data" defined in                #
#                      classes/data.py which contains the cleaned dataset.    #
#        IN/OUT PLG  - A PLG object of type "PLG" defined in classes/PLG.py.  #
#                      The PLG.statistics parameter will be updated with some #
#                      simple statistics for this map.                        #
#                                                                             #
###############################################################################
def statistics_generation(PLG_: PLG, data_: data):
    # Get the data into numpy arrays
    speed = np.array(data_.speed)
    acc = np.array(data_.acc)
    ###########################################################################
    # Speed:                                                                  #
    ###########################################################################
    # For the speed we don't want any 0 values. The cars may be stationary at
    # a traffic light and we don't want this to affect our average speed value.
    # The average speed should reflect the average speed of moving vehicles.
    non_zero_speed_data = speed[speed > 0]
    PLG_.statistics.speed_max = max(speed)
    PLG_.statistics.speed_min = min(speed)
    PLG_.statistics.speed_avg = np.average(non_zero_speed_data)
    PLG_.statistics.speed_std = np.std(non_zero_speed_data)
    ###########################################################################
    # Acceleration:                                                           #
    ###########################################################################
    PLG_.statistics.acc_max = max(acc)
    PLG_.statistics.acc_min = min(acc)
    PLG_.statistics.acc_avg = np.average(acc)
    PLG_.statistics.acc_std = np.std(acc)

    return True
