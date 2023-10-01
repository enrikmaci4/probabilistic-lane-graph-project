import numpy as np
import functions.general as g
from inputs import *
from fnames import *


ONE_FOOT_IN_METRES = 0.3048
ONE_METRE_IN_METRES = 1
# Add any other units here:

# Choose unit conversion to metres
if UNIT == "feet":
    UNIT_CONVERSION = ONE_FOOT_IN_METRES
elif UNIT == "metres":
    UNIT_CONVERSION = ONE_METRE_IN_METRES


###############################################################################
# This class will be used to load the entire original dataset.                #
#                                                                             #
# Note that not all datasets contain the "lane_id" column. Hence, we try to   #
# load this column but if it doesn't exist we will just set it to be an array #
# of zeros.                                                                   #
###############################################################################
class load_data:
    def __init__(self) -> None:
        # Load all data
        self.x = np.genfromtxt(RAW_DATA_LOC + X_NAME, dtype=float) * UNIT_CONVERSION
        self.y = np.genfromtxt(RAW_DATA_LOC + Y_NAME, dtype=float) * UNIT_CONVERSION
        self.vehicle_id = np.genfromtxt(RAW_DATA_LOC + VEHICLE_ID_NAME, dtype=int)
        try:
            self.lane_id = np.genfromtxt(RAW_DATA_LOC + LANE_ID_NAME, dtype=int)
        except FileNotFoundError:
            self.lane_id = np.zeros(len(self.x), dtype=int)
        self.time = np.genfromtxt(RAW_DATA_LOC + TIME_NAME, dtype=int)
        self.speed = np.genfromtxt(RAW_DATA_LOC + SPEED_NAME, dtype=float)
        self.acc = np.genfromtxt(RAW_DATA_LOC + ACC_NAME, dtype=float)

        # Normalise position to start at (0,0)
        self.x = self.x - min(self.x)
        self.y = self.y - min(self.y)

        # Load the sese matrices
        self.vehicle_sese = g.get_se_matrix(self.vehicle_id)
        self.lane_sese = g.get_se_matrix(self.lane_id)


###############################################################################
# This class will be used to store data. If we modify the original dataset in #
# anyway we would like a single place to store this data so we use this class #
# to store this modified data.                                                #
#                                                                             #
# The "modification" we refer to in this case is the "data cleaning" process. #
###############################################################################
class data:
    def __init__(self) -> None:
        # Instantiate the data variables
        self.num_data_points = None
        self.x = []  # x coordinate
        self.y = []  # y coordinate
        self.node = []  # node corresponding to these x,y coords
        self.lane_id = []  # lane ID
        self.vehicle_id = []  # vehicle ID
        self.time = []  # time (id)
        self.speed = []  # speed
        self.acc = []  # acceleration
        self.vehicle_sese = None  # vehicle ID sese matrix
        self.lane_sese = None  # lane ID sese matrix
