from inputs import *
import numpy as np
import cmath
import math
import classes.simulation as sim
import functions.graph as graph
import functions.general as g
from classes.PLG import *
import models.acceleration as acc_models

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
class DataRow:
    def __init__(self) -> None:
        # Initialise all values of the row to something invalid. We use a
        # character. These values will be set assigned in a separate part of
        # the program by doing something like:
        #
        # DataRow.vehicle_id = id
        #
        # instead of being passed as parameters into the constructor.
        self.vehicle_id = "x"
        self.time = "x"
        self.x = "x"
        self.y = "x"
        self.node = "x"
        self.lane_id = "x"
        self.speed = "x"
        self.acc = "x"
        self.ttc = "x"
        self.head_ang = "x"
        # Use a variable to track the number of rows in our data matrix
        self.num_rows = 9
        # Use a variable to track whether the node has changed
        self.node_changed = False


###############################################################################
# A class used to represent the vehicles in our simulation.                   #
###############################################################################
class Vehicle():
    ###########################################################################
    # Initialisation.                                                         #
    ###########################################################################
    def __init__(self, PLG: PLG, current_data: DataRow, target_destination: int) -> None:
        # Store the PLG. Objects should be passed by reference in Python so
        # this shouldn't be computationally expensive. We will not modify PLG
        # at any stage so we will never create a copy of it. We want to avoid
        # creating a copy because it is a fairly large data structure.
        self.PLG = PLG
        # Initialise the data matrix describing the trajectory for this vehicle
        self.current_state = current_data
        self.trajectory = np.zeros((0, current_data.num_rows))
        # Future nodes. This tuple stores:
        # (current node, next node, next next node)
        self.future_nodes = ()
        # We need to know this vehicle's destination
        self.target_destination = target_destination
        # List of background vehicles and their states

    ###########################################################################
    # Initialisation functions.                                               #
    ###########################################################################
    def init_future_nodes(self, start_node):
        # Generate the most likely path from the starting node to the target
        # location. We'll use the first 3 nodes in this list to populate
        # future_nodes initially.
        initial_path = graph.path_generation()

    ###########################################################################
    # Use this function to append a new row to the trajectory matrix          #
    ###########################################################################
    def append_current_data(self):
        self.trajectory = np.vstack((self.trajectory, [self.current_state.vehicle_id, self.current_state.time, self.current_state.x, self.current_state.y, self.current_state.lane_id, self.current_state.speed, self.current_state.acc, self.current_state.ttc, self.current_state.head_ang]))

    ###########################################################################
    # Functions to update the kinematics of the vehicle                       #
    ###########################################################################
    def get_acceleration(self, ttc: float):
        # Acceleration is a function of time to collision
        #self.current_state.acceleration = <function to generate acceleration from ttc>
        self.current_state.acceleration = acc_models.linear(ttc)
        return True

    def get_speed(self):
        # Use SUVAT over an interval dt.
        self.current_state.speed += sim.dt * self.current_state.acceleration
        return True
    
    def get_position(self):
        # Get the node and next node as complex numbers so
        # that we can use 2D vector algebra with these coordinates. Note that
        # this vehicle is currently traversing the edge from node to next_node
        node_pos = complex(self.PLG.nodes[self.future_nodes[0], 0], self.PLG.nodes[self.future_nodes[0], 0])
        next_node_pos = complex(self.PLG.nodes[self.future_nodes[1], 0], self.PLG.nodes[self.future_nodes[1], 0])
        next_next_node_pos = complex(self.PLG.nodes[self.future_nodes[2], 0], self.PLG.nodes[self.future_nodes[2], 0])

        # Use SUVAT over an interval dt.
        edge_phase = cmath.phase(next_node_pos - node_pos)
        ds = sim.dt * self.current_state.speed + (1/2) * (sim.dt**2) * self.current_state.acceleration
        self.current_state.x += ds * cmath.cos(edge_phase)
        self.current_state.y += ds * cmath.sin(edge_phase)

        # Now that we've updated the position, check if the node has changed.
        #
        # NOTE: A limitation of our implementation here is that ds cannot be
        #       greater than the minimum edge length, R. Therefore, the max
        #       speed should be constrained such that dt*V_max < R. This
        #       limitation is fine because if dt*V_max > R then the resolution
        #       of our simulations would become quite poor anyway.
        #
        # Convert the current position into a complex number
        current_pos = complex(self.current_state.x, self.current_state.y)

        # Get length of edge and distance we've traversed from node
        edge_length = abs(next_node_pos - node_pos)
        edge_distance_traversed = abs(current_pos - node_pos)
        
        # If we've overshot the next node, make a note of the fact that the
        # node has changed and redirect the path of the vehicle towards the new
        # next node
        if edge_distance_traversed > edge_length:
            # TODO: change node, change direction, generate new nodes
            pass
        
        return True

    def get_node(self):
        pass

    ###########################################################################
    # Function which is called to update the state of the vehicle across a    #
    # single time step.                                                       #
    ###########################################################################
    def update_kinematics(self, dt: float, ttc: float):
        # Update acceleration
        rc = self.get_acceleration(ttc)

        # Update speed
        rc = self.get_speed()

        # Update position
        rc = self.get_position()

    def step(self, dt):
        pass

    ###########################################################################
    # Functions for generating vehicle paths.                                 #
    ###########################################################################
    def get_trajectory(self):
        # We only call this function once we've arrived at a new node, so
        # current_node is now the value of next_node. We'll update
        # self.future_nodes once we've chosen the new path.
        current_node = self.future_nodes[1]

    ###########################################################################
    # Utility functions.                                                      #
    ###########################################################################
    def get_rectangle(self):
        # Intialisations
        Rx = V_LENGTH/2
        Ry = V_WIDTH/2
        alpha = self.current_state.head_ang

        # Returns a set of coordinates which describe the edges of the vehicle.
        # Note that we're modelling the vehicle as a rectangle.
        # Get the normalised coordinates
        X = g.generate_normalised_rectangle()

        # Matrix to stretch X by Rx and Ry in the x and y coords
        I_stretch = np.array([[Rx, 0],
                              [0, Ry]])

        # Get the rotation matrix
        R = np.array([[math.cos(alpha), -math.sin(alpha)],
                      [math.sin(alpha), math.cos(alpha)]])

        # NOTE: X is a tall matrix with columns [x,y]. Usually we would do M*X
        #       where M is the matrix that performs the operation we're interested
        #       in and X is a fat matrix of coordindates with rows [x]
        #                                                          [y].
        #       Since X is tall, we need to transpo.se M so we do the following
        #       matrix multiplication: X*(M^T)
        # Stretch X
        X = np.matmul(X, np.transpose(I_stretch))

        # Rotate the rectangle
        X = np.matmul(X, np.transpose(R))

        # Shift rectangle
        X[:,0] += self.current_state.x
        X[:,1] += self.current_state.y

        return X