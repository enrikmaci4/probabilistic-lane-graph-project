import sys
import os

# On my machine I need this line otherwise I get a "ModuleNotFoundError" when
# trying to import the other modules I have written within this directory.
sys.path.append(os.getcwd())

import numpy as np
from models.acceleration import A_MAX, A_MIN
from inputs import *


###############################################################################
# In this script we will store a variety of decision rules designed to choose #
# an action from a decision_list defined in the vechicle.py script.           #
###############################################################################


def rule_force_cc(decision_list: list, trajectory_length=None):
    """A rule to force a collision after around 2.5s for a 5s simulation.

        trajectory_length (int): Current trajectory length. Defaults to None.
    """
    assert trajectory_length != None
    if trajectory_length <= 100:
        return rule_2(decision_list)
    elif trajectory_length <= 150:
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
    

def _cost_dtc(dtc: float, zero_cost_threshold=5):
    """Cost function for the time to collision. We take the absolute value for 
    dtc to calculate the cost so we only need to look at the positive axis.    
    We will use a quadratic cost function which looks as follows:              
                                                                               
    C=1 _|                                                                     
         ||    |  |                                                            
         | |   | |                                                             
    C=0 _|__\ _|/____                                                          
         |     |dtc=zero_cost_threshold                                        
                                                                            
    Function description:                                                      
          --                                                                     
         | (dtc - threshold)^2                                                    
         | ------------------- for {x >= 0}{x =< threshold}                        
         |     threshold^2                                                        
    C = < 
         |          0          for {x > threshold}                                
         |
          --                                                                        

    Args:
        dtc (float): time to collision.
        zero_cost_threshold (float, optional): Threshold above which the cost
            function will return 0. Defaults to 5.
    """
    # Take the absolute value
    dtc = abs(dtc)

    # If the dtc is above the threshold return 0 instantly
    if dtc > zero_cost_threshold:
        return 0
    else:
        return ((dtc - zero_cost_threshold)/zero_cost_threshold)**2
    

def _cost_acc(da: float):
    """Penalise large changes in acceleration

    Function:
              --                     --    # da should never be above
             |       da^2              |   # A_MAX - A_MIN anyway so we should
    C = min <  -----------------  ,  1  >  # never hit the case when the cost
             | (A_MAX - A_MIN)^2       |   # function is above 1 anyway.
              --                     --    # 

    Args:
        da (float): Acceleration change.
    """
    return min((da/(A_MAX - A_MIN))**2, 1)


def _cost_speed(v: float):
    """Penalise speed if we stray too far from the average speed.

    Function:
              --                     --   
             |   (v - v_avg)^2         |  
    C = min <  -----------------  ,  1  > 
             |      v_avg^2            |  
              --                     --   

    Args:
        v (float): Acceleration change.
    """
    return min(((v - SPEED_MEAN)/SPEED_MEAN)**2, 1)


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
def _cost_5(decision):
    # Linear combination constants
    a_ttc = 0.3
    a_dtc = 0.3
    a_acc = 0.2
    a_speed = 0.2

    # Get variables of interest
    ttc = decision.ttc
    dtc = decision.dtc
    da = decision.acc - decision.prev_acc
    v = decision.speed

    # Calculate cost
    C = a_ttc*_cost_ttc(ttc) + a_dtc*_cost_dtc(dtc) + a_acc*_cost_acc(da) + a_speed*_cost_speed(v)

    return C


def rule_5(decision_list: list):
    """Rule: Minimise the cost function
    """
    # Initialise a list to store the TTC for each decision option
    L_ = []

    # Cycle through the decision options and store the TTC
    for decision_option in decision_list:
        L_.append(_cost_5(decision_option))

    # Get the path which minimises cost
    ii = np.argmin(L_)

    return decision_list[ii]