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
        ttc_list.append(abs(decision_option.ttc))

    # Get the path with the highest TTC
    ii_max_ttc = np.argmax(ttc_list)

    return decision_list[ii_max_ttc]


def rule_2(decision_list: list):
    """Rule: Choose the path with the highest DTC (lowest risk).
    """
    # Initialise a list to store the DTC for each decision option
    dtc_list = []

    # Cycle through the decision options and store the TTC
    for decision_option in decision_list:
        dtc_list.append(abs(decision_option.dtc))

    # Get the path with the highest TTC
    ii_max_dtc = np.argmax(dtc_list)

    return decision_list[ii_max_dtc]


def rule_3(decision_list: list):
    """Rule: Choose the path with the highest TTC (lowest risk).
    """
    # Initialise a list to store the TTC for each decision option
    ttc_list = []

    # Cycle through the decision options and store the TTC
    for decision_option in decision_list:
        ttc_list.append(abs(decision_option.ttc))

    # Get the path with the highest TTC
    ii_max_ttc = np.argmin(ttc_list)

    return decision_list[ii_max_ttc]


def rule_4(decision_list: list):
    """Rule: Choose the path with the highest DTC (lowest risk).
    """
    # Initialise a list to store the DTC for each decision option
    dtc_list = []

    # Cycle through the decision options and store the TTC
    for decision_option in decision_list:
        dtc_list.append(abs(decision_option.dtc))

    # Get the path with the highest TTC
    ii_max_dtc = np.argmin(dtc_list)

    return decision_list[ii_max_dtc]