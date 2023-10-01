from inputs import *
from fnames import *

###############################################################################
# Filepaths: Data structures                                                  #
#                                                                             #
# RAW_DATA_LOC   - Location of the raw data. This folder should contain 7     #
#                  files with the following information: accelerations, time  #
#                  x, y, lane ID, speed, vehicle ID.                          #
#                                                                             #
#                  FILE NAMES:                                                #
#                  - ACC_NAME        = "Acceleration"                         #
#                  - TIME_NAME       = "Frame_ID"                             #
#                  - X_NAME          = "Global_X"                             #
#                  - Y_NAME          = "Global_Y"                             #
#                  - LANE_ID_NAME    = "Lane_ID"                              #
#                  - SPEED_NAME      = "Speed"                                #
#                  - VEHICLE_ID_NAME = "Vehicle_ID"                           #
#                                                                             #
# CLEAN_DATA_LOC - The raw data is processed and saved to this directory.     #
#                                                                             #
#                  FILE NAMES:                                                #
#                  - CLEAN_DATA_NAME = "clean_data"                           #
#                                                                             #
# PLG_SAVE_LOC   - The PLG data structure is saved im this location.          #
#                                                                             #
#                  FILE NAMES                                                 #
#                  - PLG             = "PLG"                                  #
#                                                                             #
# #############################################################################
RAW_DATA_LOC = "data/" + DATASET + "/original/"
CLEAN_DATA_LOC = "data/" + DATASET + "/clean/"
PLG_SAVE_LOC = "data/" + DATASET + "/structs/"

ACC_NAME = "Acceleration"
TIME_NAME = "Frame_ID"
X_NAME = "Global_X"
Y_NAME = "Global_Y"
LANE_ID_NAME = "Lane_ID"
SPEED_NAME = "Speed"
VEHICLE_ID_NAME = "Vehicle_ID"

CLEAN_DATA_NAME = "clean_data"

PLG_NAME = "PLG"

###############################################################################
# Filenames: Simulation outputs                                               #
#                                                                             #
# TEST_SIM_SAVE_LOC                                                           #
#                - A directory where we save single animations/data           #
#                  structures for testing purposes. I.e. the output from      #
#                  test_simulation_1.py will be saved here. This script is    #
#                  typically used to generate a single animation to evaluate  #
#                  (by eye) minor changes to the models.                      #
#                                                                             #
#                  FILE NAMES:                                                #
#                  - Described below.                                         #
#                                                                             #
# SET(1/2/3)_SAVE_LOC                                                         #
#                - When we run automated simulations (i.e. run 100            #
#                  simulations) to gauge the effectiveness of a method across #
#                  a large number of attempts, the outputs will be classified #
#                  and saved to this directory.                               #
#                - SET1/2: We plan to use two different models. One which is  #
#                  designed to drive "safely" and avoid collisions and one    #
#                  which is optimised to generate realistic corner cases. The #
#                  outputs from the "safe" model will be in SET1 and the      #
#                  outputs from the "corner case" model will be in SET 2.     #
#                - SET3: Is similar to set1 but it will contain funky cases   #
#                  where the initial state is customised (set by the          #
#                  designer) rather than randomly generated.                  #
#                                                                             #
#                  FILE NAMES:                                                #
#                  - SIM_ANIM_NAME   = "anim_ii". Saved as a GIF.             #
#                  - SIM_DATA_PKL_NAME                                        #
#                                    = "simdata_pkl_ii". The data is saved as #
#                                      a Python pickle. The data is a Python  #
#                                      list of Vehicle objects (defined in    #
#                                      vehicle.py). Each Vehicle object       #
#                                      should have the same length            #
#                                      self.trajectory matrix.                #
#                  - SIM_DATA_TXT_NAME                                        #
#                                    = "simdata_txt_ii". Saved in plain text  #
#                                      format. The data is a 2D big data      #
#                                      matrix describing the simulation. The  #
#                                      matrix is every vehicle's trajectory   #
#                                      matrices concatenated on top of each   #
#                                      other.                                 #
#                  - STATS_NAME      = "stats". Write some statistics about   #
#                                      this simulation run.                   #
#                                                                             #
#                  FILE NAME SUFFIXES:                                        #
#                  - (N)CC           = "_cc/_ncc". cc are the scenrios which  #
#                                      terminate with a collision and ncc do  #
#                                      not (where ncc is "no corner case").   #
#                  - IS              = "_is". If a file ends in "is" then it  #
#                                      is an initial state file used to seed  #
#                                      a simulation.                          #
#                                                                             #
#                  - NOTE: ii is the simulation index and it will be appended #
#                    to the save name by the script which is saving the       #
#                    files. If we're not generating many simulations and      #
#                    we're only generating a single simulation for test       #
#                    purposes then these files with be saved in               #
#                    TEST_SIM_SAVE_LOC without the "ii" part. I.e. just       #
#                    "anim" for the GIF.                                      #
#                                                                             #
# #############################################################################
TEST_SIM_SAVE_LOC = "output/test/"
SET1_SAVE_LOC = "output/set1/"
SET2_SAVE_LOC = "output/set2/"
SET3_SAVE_LOC = "output/set3/"

SIM_ANIM_NAME = "anim"
SIM_DATA_PKL_NAME = "simdata_pkl"
SIM_DATA_TXT_NAME = "simdata_txt"
STATS_NAME = "stats"

CC_SUFF = "_cc"
NCC_SUFF = "_ncc"
IS_SUFF = "_is"
