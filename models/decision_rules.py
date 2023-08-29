import sys
import os

# On my machine I need this line otherwise I get a "ModuleNotFoundError" when
# trying to import the other modules I have written within this directory.
sys.path.append(os.getcwd())

import numpy as np
from models.acceleration import A_MAX, A_MIN
from inputs import *
import functions.graph as graph
import random


###############################################################################
# In this script we will store a variety of decision rules designed to choose #
# an action from a decision_list defined in the vechicle.py script.           #
###############################################################################


def rule_force_cc(decision_list: list, trajectory_length=None):
    """A rule to force a collision after around 2.5s for a 5s simulation.

        trajectory_length (int): Current trajectory length. Defaults to None.
    """
    assert trajectory_length != None
    if trajectory_length <= 50:
        return rule_2(decision_list)
    elif trajectory_length <= 75:
        return rule_3(decision_list)
    else:
        return rule_4(decision_list)
    

def rule_force_cc_no_lane_change(decision_list: list, trajectory_length=None):
    """A rule to force a collision 1D collisions.
    """
    assert trajectory_length != None
    for decision in decision_list:
        decision.acc = random.uniform(0, 1)
    if trajectory_length <= 50:
        return rule_2(decision_list)
    elif trajectory_length <= 75:
        return rule_3(decision_list)
    else:
        return rule_4(decision_list)


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


###############################################################################
# Cost functions                                                              #
#                                                                             #
# NOTE: - All cost functions will be normalised between 0 and 1.              #
#       - 0 is a low cost/desirable action and 1 is a high cost/undesirable   #
#         action.                                                             #
#       - We use convex programming for optimisation so write down convex     #
#         cost functions which we know have a global minimum.                 #
#                                                                             #
###############################################################################
def _cost_ttc(ttc: float, zero_cost_threshold=5):
    """Cost function for the time to collision. We take the absolute value for 
    ttc to calculate the cost so we only need to look at the positive axis.    
    We will use a quadratic cost function which looks as follows:              
                                                                               
    C=1 _|                                                                     
         ||    |  |                                                            
         | |   | |                                                             
    C=0 _|__\ _|/____                                                          
         |     |ttc=zero_cost_threshold                                        
                                                                            
    Function description:                                                      
          --                                                                     
         | (ttc - threshold)^2                                                    
         | ------------------- for {x >= 0}{x =< threshold}                        
         |     threshold^2                                                        
    C = < 
         |          0          for {x > threshold}                                
         |
          --                                                                        

    Args:
        ttc (float): time to collision.
        zero_cost_threshold (float, optional): Threshold above which the cost
            function will return 0. Defaults to 5.
    """
    # Take the absolute value
    ttc = abs(ttc)

    # If the ttc is above the threshold return 0 instantly
    if ttc > zero_cost_threshold:
        return 0
    else:
        return ((ttc - zero_cost_threshold)/zero_cost_threshold)**2
    

def _cost_dtc(dtc: float, zero_cost_threshold=20):
    """Cost function for the time to collision. We take the absolute value for 
    dtc to calculate the cost so we only need to look at the positive axis.    
    We will use a quadratic cost function which looks as follows:              
                                                                               
    C=1 _|__                                                                      
         |  ||    |  |                                                            
         |  | |   | |                                                             
    C=0 _|__|__\__|/____                                                          
         |  |dD   |dtc=zero_cost_threshold                                        
                                                                            
    Function description:                                                      
          --                                                                     
         | (dtc - (threshold + dD))^2                                                    
         | --------------------------    for {x >= 0}{x =< threshold}                        
         |         threshold^2                                                        
    C = < 
         |              0                for {x > threshold}                                
         |
          --                                                                        

    Args:
        dtc (float): time to collision.
        zero_cost_threshold (float, optional): Threshold above which the cost
            function will return 0. Defaults to 5.
    """
    # Take the absolute value
    dtc = abs(dtc)
    dD = 5

    # If the dtc is above the threshold return 0 instantly
    if dtc > zero_cost_threshold + dD:
        return 0
    else:
        return ((dtc - (zero_cost_threshold + dD))/zero_cost_threshold)**2
    

def _cost_acc(da: float):
    """Penalise large changes in acceleration

    Function:
           
              da^2           
    C = -----------------  
        (A_MAX - A_MIN)^2            

    Args:
        da (float): Acceleration change.
    """
    return (da/(A_MAX - A_MIN))**2


def _cost_speed(v: float):
    """Penalise speed if we stray too far from the average speed.

    Function:
                        
          (v - v_avg)^2  
    C = -----------------
            v_avg^2     
                        

    Args:
        v (float): Acceleration change.
    """
    return ((v - SPEED_MEAN)/SPEED_MEAN)**2


def _cost_lane_changes(decision, PLG_=None):
    """Penalise if we keep zig-zagging between lanes.

    Function:
          N
    C = -----
         N_L

    Args:
        decision type data structure defined in vehicle.py.
    """
    assert decision.current_trajectory_length != graph.EMPTY_ENTRY
    # Check if the trajectory is the correct length, for simplicity we don't
    # start computing this cost until the trajectory length is atleast N_L.
    # Get the N_L
    if decision.current_trajectory_length >= decision.N_L:
        # Get N_L previous nodes
        path = list(decision.N_L_prev_path)
        # Append 1 next_node
        path.append(decision.path[1])

        # Return cost
        return graph.calculate_num_lane_changes(PLG_, path)/decision.N_L
    else:
        return 0


###############################################################################
# Pre-optimisation:                                                           #
#                                                                             #
# Minimise a linear combination of cost functions.                            #
#                                                                             #
#     min { a1*cost_ttc + a2*cost_dtc + a3*cost_acc + a4*cost_speed }         #
#                                                                             # 
# Where:                                                                      #
#                                                                             #
#    a1 + a2 + a3 + a3 = 1                                                    #
#                                                                             #
#    ai >= 0 for i = 1,2,3,4                                                  #
#                                                                             #
###############################################################################
def _cost_5(decision, PLG_=None):
    # Linear combination constants
    a_ttc = 0.2
    a_dtc = 0.4
    a_acc = 0.1
    a_speed = 0.1
    a_lane_change = 0.1

    # Get variables of interest
    ttc = abs(decision.ttc)
    dtc = abs(decision.dtc)
    da = decision.acc - decision.prev_acc
    v = decision.speed

    # Calculate cost
    C = a_ttc*_cost_ttc(ttc) + \
        a_dtc*_cost_dtc(dtc) + \
        a_acc*_cost_acc(da) + \
        a_speed*_cost_speed(v) + \
        a_lane_change*_cost_lane_changes(decision, PLG_=PLG_)

    return C


def rule_5(decision_list: list, PLG_=None):
    """Rule: Minimise the cost function
    """
    # Initialise a list to store the TTC for each decision option
    L_ = []

    # Cycle through the decision options and store the TTC
    for decision_option in decision_list:
        L_.append(_cost_5(decision_option, PLG_=PLG_))

    # Get the path which minimises cost
    ii = np.argmin(L_)

    return decision_list[ii]