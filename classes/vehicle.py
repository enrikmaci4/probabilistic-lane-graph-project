
import numpy as np
import classes.simulation as sim

###############################################################################
# Create a data structure to store the information for one row of our data    #
# matrix. The columns of the data matrix will be as follows:                  #
#                                                                             #
# 1.  Vehicle ID                                                              #
# 2.  Time (either Frame ID or in sec)                                        #
# 3.  x position                                                              #
# 4.  y position                                                              #
# 5.  Node ID                                                                 #
# 6.  Lane ID                                                                 #
# 7.  Speed                                                                   #
# 8.  Acceleration                                                            #
# 9.  Time to collision (TTC) in secconds                                     #
# 10. Heading angle (in radians)                                              #
#                                                                             #
# The data matrix describing the trajectory of one vehicle will be a set of   #
# rows described above concatenated into a 2D matrix. Let the data matrix     #
# describing the trajectory for the i'th vehicle be D_i.                      #
#                                                                             #
# A full simulation which describes the trajectories and interactions between #
# multiple vehicles will then be described by a set of the matrices described #
# above which are concatenated, one on top of another. E.g. a simulation with #
# n vehicles will be described by the matrix:                                 #
#                                                                             #
# Simulation Matrix = [D_1,                                                   #
#                      D_2,                                                   #
#                      ...,                                                   #
#                      D_0]                                                   #
#                                                                             #
###############################################################################
class DataRow():
    def __init__(self) -> None:
        # Initialise all values of the row to something invalid. We use a
        # character. These values will be set assigned in a separate part of
        # the program by doing something like:
        #
        # DataRow.vehicle_id = id
        #
        # instead of being passed as parameters into the constructor.
        vehicle_id = "x"
        time = "x"
        x = "x"
        y = "x"
        node_id = "x"
        lane_id = "x"
        speed = "x"
        acc = "x"
        ttc = "x"
        head_ang = "x"
        # Use a variable to track the number of rows in our data matrix
        num_rows = 9


###############################################################################
# Create a data structure to store the current node, next node, and next next #
# node.                                                                       #
###############################################################################


###############################################################################
# A class used to represent the vehicles in our simulation.                   #
###############################################################################
class Vehicle():
    ###########################################################################
    # Initialisation.                                                         #
    ###########################################################################
    def __init__(self, current_data: DataRow) -> None:
        # Initialise the data matrix describing the trajectory for this vehicle
        self.current_data = current_data
        self.trajectory_matrix = np.zeros((0, current_data.num_rows))
        # Future nodes. This tuple stores the current node, next node and next
        # next node in this vehicle's path.
        self.future_nodes = ()


    ###########################################################################   
    # Use this function to append a new row to the trajectory matrix          #
    ###########################################################################
    def append_current_data(self):
        self.trajectory_matrix = np.vstack((self.trajectory_matrix, [self.current_data.vehicle_id, self.current_data.time, self.current_data.x, self.current_data.y, self.current_data.lane_id, self.current_data.speed, self.current_data.acc, self.current_data.ttc, self.current_data.head_ang]))

    ###########################################################################
    # Functions to update the kinematics of the vehicle                       #
    ###########################################################################
    def get_acceleration(self, ttc):
        # Acceleration is a function of time to collision
        #self.current_data.acceleration = <function to generate acceleration from ttc>
        pass

    def get_speed(self):
        # Use SUVAT over an interval dt.
        self.current_data.speed += sim.dt * self.current_data.acceleration
    
    def get_position(self):
        # Use SUVAT over an interval dt.
        ds = sim.dt * self.current_data.speed + (1/2) * (sim.dt**2) * self.current_data.acceleration

        self.current_data.x += sim.dt * self.current_data.speed + (1/2) * (sim.dt**2) * self.current_data.acceleration
        self.current_data.y += sim.dt * self.current_data.speed + (1/2) * (sim.dt**2) * self.current_data.acceleration

    def get_node(self):
        pass