import numpy as np
from classes.vehicle import *


###############################################################################
# In this script we will store a variety of decision rules designed to choose #
# an action from a decision_list defined in the vechicle.py script.           #
###############################################################################


def rule_1(decision_list: list):
    """Rule: Choose the path with the highest TTC (lowest risk).
    """
    # Initialise a list to store the TTC for each decision option
    ttc_list = []

    # Cycle through the decision options and store the TTC
    for decision_option in decision_list:
        ttc_list.append(decision_option.ttc)

    # Get the path with the highest TTC
    ii_max_ttc = np.argmax(ttc_list)

    return decision_list[ii_max_ttc]


