from inputs import *
from fnames import *
import numpy as np
import math
import cmath
import functions.graph as graph
import functions.general as g
from classes.PLG import *
import models.acceleration as acc_models
import models.decision_rules as rules
import random

NUM_COLS_IN_DATA_MATRIX = 11
# TRAJECTORY (DATA MATRIX) INDICES
II_VEHICLE_ID = 0
II_TIME = 1
II_NODE = 2
II_X = 3
II_Y = 4
II_LANE_ID = 5
II_SPEED = 6
II_ACC = 7
II_TTC = 8
II_DTC = 9
II_HEAD_ANG = 10
# SIGNALS
SIGNAL_CONTINUE_SIM = 0
SIGNAL_TERM_SIM = 1
SIGNAL_COLLISION = 2

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
        self.vehicle_id = graph.EMPTY_ENTRY
        self.time = graph.EMPTY_ENTRY
        self.x = graph.EMPTY_ENTRY
        self.y = graph.EMPTY_ENTRY
        self.node = graph.EMPTY_ENTRY
        self.lane_id = graph.EMPTY_ENTRY
        self.speed = graph.EMPTY_ENTRY
        self.acc = graph.EMPTY_ENTRY
        self.ttc = graph.EMPTY_ENTRY
        self.dtc = graph.EMPTY_ENTRY
        self.head_ang = graph.EMPTY_ENTRY
        # Store the most likely path here so that we can use it as a
        # "prediction" for the motion of background vehicls
        self.most_likely_path = graph.EMPTY_ENTRY


###############################################################################
# Create a data structure to store the information we need to make a          #
# "decision". A decision here is choosing a trajectory and an acceleration.   #
#                                                                             #
###############################################################################
class Decision:
    def __init__(self) -> None:
        # Initialise all values of the row to something invalid. We use a
        # character. These values will be set assigned in a separate part of
        # the program.
        self.path = graph.EMPTY_ENTRY
        self.ttc = graph.EMPTY_ENTRY
        self.dtc = graph.EMPTY_ENTRY
        self.prev_acc = graph.EMPTY_ENTRY
        self.acc = graph.EMPTY_ENTRY
        self.speed = graph.EMPTY_ENTRY
        self.num_lane_changes_in_path = graph.EMPTY_ENTRY


###############################################################################
# A class used to represent the vehicles in our simulation.                   #
###############################################################################
class Vehicle:
    ###########################################################################
    # Initialisation.                                                         #
    ###########################################################################
    def __init__(self, PLG: PLG, current_data: DataRow, target_destination: int) -> None:
        # We should never generate a list of vehicles which are already in a
        # collided state so initialise this to False. If we detect a collision,
        # set this to True.
        self.is_collision = False
        # Store the PLG. Objects should be passed by reference in Python so
        # this shouldn't be computationally expensive. We will not modify PLG
        # at any stage so we will never create a copy of it. We want to avoid
        # creating a copy because it is a fairly large data structure.
        self.PLG = PLG
        # We need to know this vehicle's destination
        self.target_destination = target_destination
        # Initialise the current data
        self.current_state = current_data
        self.current_state.ttc = graph.INF
        self.init_most_likely_path()
        # Initialise the matrix describing the trajectory for this vehicle
        self.trajectory = np.zeros((0, NUM_COLS_IN_DATA_MATRIX))
        self.append_current_data()
        self.trajectory_length = 1
        # Future nodes. This tuple stores:
        # (current node, next node)
        assert self.current_state.most_likely_path[0] == current_data.node
        self.future_nodes = [current_data.node, self.current_state.most_likely_path[1]]
        # List of background vehicles and their states
        self.bv_list = []
        # List of possible decisions
        self.decision_list = []
        # Current decision - we'll choose from "decision_list" and assign 
        # outputs of this decision to self.current_State and self.future_nodes.
        self.decision = Decision()
        # If overshoot > 0, this means that we have arrived at a new node.
        # Furthermore, this also stores the overshoot distance, once we know
        # what our next node is we can traverse this distance in the direction
        # of that node.
        self.overshoot = 0
        # Force CC - Set this to True if you want to forcefully generate a
        # collision. This option is only compatible with 5 second simulation
        # lengths right now.
        self._force_cc = False

    ###########################################################################
    # Initialisation functions.                                               #
    ###########################################################################
    def init_most_likely_path(self):
        self.current_state.most_likely_path = graph.path_generation(self.PLG, self.current_state.node, self.target_destination)
        return True

    ###########################################################################
    # Use this function to append a new row to the trajectory matrix          #
    ###########################################################################
    def append_current_data(self):
        self.trajectory = np.vstack((self.trajectory, [
            self.current_state.vehicle_id,      # 0 II_VEHICLE_ID
            self.current_state.time,            # 1 II_TIME
            self.current_state.node,            # 2 II_NODE
            self.current_state.x,               # 3 II_X
            self.current_state.y,               # 4 II_Y
            self.current_state.lane_id,         # 5 II_LANE_ID
            self.current_state.speed,           # 6 II_SPEED
            self.current_state.acc,             # 7 II_ACC
            self.current_state.ttc,             # 8 II_TTC
            self.current_state.dtc,             # 9 II_DTC
            self.current_state.head_ang         # 10 II_HEAD_ANG
            ]))

    ###########################################################################
    # Functions to update the kinematics of the vehicle                       #
    ###########################################################################
    def update_speed(self):
        # Use SUVAT over an interval dt.
        self.current_state.speed += dt * self.current_state.acc
        #if self.current_state.speed >= 10:
        #    self.current_state.speed = 10 + random.uniform(-2, 2)
        return True
    
    def update_position(self):
        # Get the node and next node as complex numbers so
        # that we can use 2D vector algebra with these coordinates. Note that
        # this vehicle is currently traversing the edge from node to next_node
        assert self.future_nodes[0] != self.future_nodes[1]
        node_pos = complex(self.PLG.nodes[self.future_nodes[0], 0], self.PLG.nodes[self.future_nodes[0], 1])
        next_node_pos = complex(self.PLG.nodes[self.future_nodes[1], 0], self.PLG.nodes[self.future_nodes[1], 1])

        # Use SUVAT over an interval dt.
        edge_phase = cmath.phase(next_node_pos - node_pos)
        ds = dt * self.current_state.speed + (1/2) * (dt**2) * self.current_state.acc
        self.current_state.x += ds * math.cos(edge_phase)
        self.current_state.y += ds * math.sin(edge_phase)

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
            # We've arrived at a new node, actions to do here:
            # - Store the amount by which we've overshot the edge.
            # - Reset the current position to the location of the new node.
            self.next_node_overshoot = edge_distance_traversed - edge_length
            self.current_state.x = self.PLG.nodes[self.future_nodes[1], 0]
            self.current_state.y = self.PLG.nodes[self.future_nodes[1], 1]
            self.overshoot = edge_distance_traversed - edge_length
        
        return True

    def add_overshoot(self):
        """This function should be called once self.future_nodes has been
        updated with the new next node.
        """
        # Check that overshoot is non-zero
        assert self.overshoot > 0

        # Get the node and next node as complex numbers so
        # that we can use 2D vector algebra with these coordinates. Note that
        # this vehicle is currently traversing the edge from node to next_node
        assert self.future_nodes[0] != self.future_nodes[1]
        node_pos = complex(self.PLG.nodes[self.future_nodes[0], 0], self.PLG.nodes[self.future_nodes[0], 0])
        next_node_pos = complex(self.PLG.nodes[self.future_nodes[1], 0], self.PLG.nodes[self.future_nodes[1], 0])

        # Use SUVAT over an interval dt.
        edge_phase = cmath.phase(next_node_pos - node_pos)
        self.current_state.x += self.overshoot * math.cos(edge_phase)
        self.current_state.y += self.overshoot * math.sin(edge_phase)

        # Reset the overshoot to zero
        self.overshoot = 0

        return True

    ###########################################################################
    # Function which is called to update the state of the vehicle across a    #
    # single time step.                                                       #
    ###########################################################################
    def update_kinematics(self):
        # Update speed
        rc = self.update_speed()

        # Update position
        rc = self.update_position()

    def step(self, ii: int, v_list: list):
        # ii     - The time frame.
        # v_list - The complete list of vehicles in this simulation.
        #
        # Let's say that the following is a set of nodes in the PLG. We will
        # classify the position of the vehicle into two different states:
        # - State #1: A vehicle is currently positioned on an edge.
        # - State #2: A vehicle is currently positioned on a node.
        #              o
        #       o     /|
        #        \   / |
        #         \ /  |
        #          o---o <-- A vehicle positioned here: state #2
        #         / \  |
        #        /   \ | <-- A vehicle positioned here: state #1
        #       o     \|
        #              o
        #
        # Distiniguishing between these two states is important because:
        # - In state #1 the vehicle cannot change it's trajectory. It is
        #   confined to travel along the direction of the edge until it reaches
        #   a node. This means that the actions we need to generate in response
        #   to an input feature vector is {acc}. We cannot modify the
        #   trajectory of the vehicle anyway so there is no point trying to
        #   generate a new trajectory in this region, especially when
        #   generating trajectories is the most expensive computation.
        # - In state #2, the vehicle may modify both it's trajectory and
        #   acceleration. There is a choice of multiple next nodes therefore we
        #   need to geneerate, both, a trajectory and an acceleration in
        #   response to the input feature vector: {acc, path}.
        #
        # Generating a trajectory is the more expentive computation therefore
        # this kind of model is helpful for us to simplify the cost of
        # simulation. If we create, say 10 second simulation, and in those 10s
        # a vehicle traverses a total of 5 different nodes, we only need to 
        # the more expensive computation 5 times instead of at every single
        # time step.

        # Initialisations
        rc_signal = SIGNAL_CONTINUE_SIM
        current_node = self.future_nodes[0]
        overshot_edge = False

        # Update the kinematics of the vehicle based on the previously
        # generated acceleration. We need to update them first thing because if
        # there's an overshoot we need to catch it straight away and decide
        # the new trajectory. Note that we're using the acceleration that we
        # generated in the previous time step. This makes sense intuitively
        # because todays acceleration affects tomorrow's state. This is
        # essentially an implementeation of that logic...
        rc = self.update_kinematics()

        # First update the list of background vehicles kinematics of 
        # vehicle.
        rc = self.bv_detection(v_list)

        # Now we have a list of background vehicles, we can proceed to
        # generating our own actions based on what's going on around us. First,
        # we need to know our TTC with each vehicle.
        # - If we're generating a new path, we need to calculate the TTC to
        #   every vehicle for every path and then choose a path given this
        #   information.
        # - If we're not generating a new path, we can calculate the TTC 
        #   every vehicle given our current path and the current most likely
        #   path of each vehicle.
        if self.overshoot > 0:
            # Intialisations
            # - Remember that the node has changes so we need to set the
            #   "current node" to the previous "next node".
            current_node  = self.future_nodes[1]
            overshot_edge = True

            # Check if we've reached the target destination
            if current_node in self.PLG.target_clusters[self.target_destination]:
                rc_signal = SIGNAL_TERM_SIM
            
            # First generate a tree of paths
            path_tree = graph.fast_path_tree_generation(self.PLG, current_node, self.target_destination)

            # Create a list of possible decisions
            self.decision_list = []
            for ii in path_tree:
                # Initialse
                path = path_tree[ii]
                decision_option = Decision()
                ttc = graph.INF
                dtc = graph.INF
                
                # Now calculate the TTC between this path and the background
                # vehicles. We will take the minimum TTC as this is what will
                # have the highest risk.
                #
                # NOTE: We are ignoring negative TTCs for now because we
                # shouldn't really be responding to vehicles speeding behind us
                # anyway. That's a bit dangerous...
                for bv in self.bv_list:
                    # Calculate TTC to this BV
                    ttc_for_this_bv, dtc_for_this_bv = graph.calculate_ttc_and_dtc(self.PLG, path, self.current_state.speed, self.current_state.acc, bv.most_likely_path, bv.speed, bv.acc)

                    if (abs(ttc_for_this_bv) < abs(ttc)):
                        ttc = ttc_for_this_bv

                    if (abs(dtc_for_this_bv) < abs(dtc)):
                        dtc = dtc_for_this_bv

                # Now calculate/set the rest of decision data
                decision_option.path = path
                decision_option.ttc = ttc
                decision_option.dtc = dtc
                decision_option.num_lane_changes_in_path = graph.calculate_num_lane_changes(self.PLG, path)
                decision_option.prev_acc = self.current_state.acc
                decision_option.acc = acc_models.linear(ttc=ttc, dtc=dtc)
                decision_option.speed = self.current_state.speed + dt*decision_option.acc

                # Append this decision_option to the list of possible decisions
                self.decision_list.append(decision_option)

            # Now choose an action from the list of possible decisions
            if self._force_cc:
                self.decision = rules.rule_force_cc(self.decision_list, trajectory_length=self.trajectory_length)
            else:
                self.decision = rules.rule_5(self.decision_list)

            # Update the path in the current state
            self.current_state.most_likely_path = self.decision.path

            # Update self.future_nodes
            self.future_nodes[0] = current_node
            try:
                # Try to get the next node, however, this can fail if we
                # generate a path of length 1
                self.future_nodes[1] = self.current_state.most_likely_path[1]
            except IndexError:
                # If we've generated a path of length 1, handle this gracefully
                # by returning SIGNAL_TERM_SIM to terminate the simulation
                rc_signal = SIGNAL_TERM_SIM

            # Don't forget to add the overshoot!
            self.add_overshoot()

        else:
            # Calculate TTC
            ttc = graph.INF
            dtc = graph.INF
            path = self.current_state.most_likely_path

            # Cycle through the BVs
            for bv in self.bv_list:
                # Calculate TTC to this BV
                ttc_for_this_bv, dtc_for_this_bv = graph.calculate_ttc_and_dtc(self.PLG, path, self.current_state.speed, self.current_state.acc, bv.most_likely_path, bv.speed, bv.acc)

                if (abs(ttc_for_this_bv) < abs(ttc)):
                    ttc = ttc_for_this_bv

                if (abs(dtc_for_this_bv) < abs(dtc)):
                    dtc = dtc_for_this_bv

            # Update the information in our decision - the informaiton in the
            # decision struct will then be propagated onto the current_state.
            self.decision.ttc = ttc
            self.decision.dtc = dtc
            self.decision.acc = acc_models.linear(ttc=ttc, dtc=dtc)

        # Update the current state information
        self.current_state.ttc = self.decision.ttc
        self.current_state.dtc = self.decision.dtc
        self.current_state.acc = self.decision.acc
        self.current_state.time = ii*dt
        self.current_state.node = current_node

        # Append this data row to the trajectory matrix
        self.append_current_data()
        if overshot_edge:
            self.trajectory[-1, II_HEAD_ANG] = self.trajectory[-2, II_HEAD_ANG]
        else:
            self.trajectory[-1, II_HEAD_ANG] = self._get_head_ang_2()
        self.trajectory_length += 1

        # NOTE: I've figured out why the heading angle is jerky! Consider the
        #       following node set up:
        #
        #           x3    x4         If we calculate the current heading angle
        # (node) x2 o--.----.----o   as the phase of (dx,dy). Then we have the
        #          /                 following cases:
        #     x1  .                  
        #        /                   Case 1 - Travelling from x1->x2 OR x3->x4.
        #       /                    This results in a heading angle which is
        #      o                     perfectly aligned along the edge, this is
        #                            what we want.
        #
        #                            Case 2 - The vehicle overshoots its
        #                            current edge, e.g., it goes from x1->x3.
        #                            in this case we get a heading angle which
        #                            is not aligned with either the first edge
        #                            or the second edge.
        #
        # How do we solve this? When we detect that we've overshot an edge,
        # repeat the same heading angle as before. Then, on the next time step
        # the heading angle calculation will yield the desired result.

        return rc_signal

    ###########################################################################
    # Utility functions.                                                      #
    ###########################################################################
    def get_rectangle(self, ii=None, x_scale=1, y_scale=1):
        # Intialisations
        Rx = V_LENGTH/2
        Ry = V_WIDTH/2
        if ii == None:
            x = self.current_state.x
            y = self.current_state.y
            alpha = self.current_state.head_ang
        else:
            x = self.trajectory[ii,II_X]
            y = self.trajectory[ii,II_Y]
            alpha = self.trajectory[ii,II_HEAD_ANG]

        # Returns a set of coordinates which describe the edges of the vehicle.
        # Note that we're modelling the vehicle as a rectangle.
        # Get the normalised coordinates
        X = g.generate_normalised_rectangle()

        # Matrix to stretch X by Rx and Ry in the x and y coords
        I_stretch = np.array([[x_scale*Rx, 0],
                              [0, y_scale*Ry]])

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
        X[:,0] += x
        X[:,1] += y

        return X
    
    def bv_detection(self, v_list: list):
        """A function which detects the number of BVs around and stores their
        current state. This state will be used to calculate the TTC with that
        vehicle.
        
        Args:
            v_list (list): List of vehicle objects.
        """
        # Initialisations
        xc = self.current_state.x
        yc = self.current_state.y
        alpha = self.current_state.head_ang
        Rx = BV_DETECTION_RX
        Ry = BV_DETECTION_RY
        self.v_list = []

        # Iterate over the vehicle list and store the state of the vehicles.
        for V in v_list:
            if V.current_state.vehicle_id != self.current_state.vehicle_id:
                # Current node of this BV
                bv_x = V.current_state.x
                bv_y = V.current_state.y

                # Check if this is within the BV detection zone
                is_bv = g.is_in_rectangle(bv_x, bv_y, xc, yc, alpha, Rx, Ry)

                # If this is a BV then store its state in self.bv_list
                if is_bv:
                    self.bv_list.append(V.current_state)

        return True

    def _get_head_ang_1(self):
        """Calculate the most recent phase using the x,y coordinates in the
        trajectory.

        NOTE: The trajectory length should always be ATLEAST 1 before this
        function is called.
        """
        # Initialisations
        mov_avg_win = 2

        # Get node coords
        node_path = self.trajectory[:,II_NODE]
        unique_node_path = [node_path[0]]
        path_length = len(node_path)

        # Unique-ify the node path
        for ii in range(path_length-1):
            next_node = node_path[ii+1]
            # Only append this node if it is different to the previoud node
            if next_node != unique_node_path[-1]:
                unique_node_path.append(next_node)

        # Append some of the future node since we're travelling in that
        # direction the car should orient itself in that direction
        #
        # NOTE: We could get an error here if the path length is only 1 or 2
        #       long
        if unique_node_path[-1] != self.current_state.most_likely_path[1]:
            unique_node_path.append(self.current_state.most_likely_path[1])

        #if unique_node_path[-1] != self.current_state.most_likely_path[2]:
        #    unique_node_path.append(self.current_state.most_likely_path[2])

        # Phase list
        num_nodes_in_path = len(unique_node_path)

        # Check the number of nodes in the path
        assert num_nodes_in_path >= 1
        if num_nodes_in_path == 1:
            # Just return the current value
            return self.current_state.head_ang
        
        elif num_nodes_in_path-1 < mov_avg_win:
            # NOTE: We use num_nodes_in_path-1 because if there are N nodes
            #       there are N-1 phases.
            # Turn the node path into a phase list
            phase_list = graph.node_list_to_edge_phase(self.PLG, unique_node_path)
            return np.average(phase_list)
        
        else:
            # Turn the node path into a phase list and take the average of the
            # last mov_avg_win number of phases
            phase_list = graph.node_list_to_edge_phase(self.PLG, unique_node_path)
            return np.average(phase_list[-mov_avg_win::])

    def _get_head_ang_2(self):
        """Simple model: Use just the current and previous x,y coords and we
        will smooth the result after the simulation is finished
        """
        # Get the current x,y coords
        x_curr = self.trajectory[-1,II_X]
        y_curr = self.trajectory[-1,II_Y]

        # Previous x,y coords
        x_prev = self.trajectory[-2,II_X]
        y_prev = self.trajectory[-2,II_Y]

        # dx and dy
        dx = x_curr - x_prev
        dy = y_curr - y_prev

        # Check the number of nodes in the path
        if  (dx == 0) and (dy == 0):
            # Just return the current value
            return self.current_state.head_ang

        else:
            # Turn the node path into a phase list and take the average of the
            # last mov_avg_win number of phases
            return cmath.phase(complex(dx, dy))


def get_min_max_dtc(V: Vehicle, min_or_max=None):
    """Get the minimum/maxiumum DTC for a vehicle for it's current decision
    list.

    Args:
        V (Vehicle): Vehicle
        min_or_max (string): either "min" or "max".
    """
    # Initialisations
    dtc_list = []

    # Get dtc as a list
    for decision in V.decision_list:
        dtc_list.append(decision.dtc)

    # Return either the min or max
    if min_or_max=="max":
        return max(dtc_list)
    elif min_or_max == "min":
        return min(dtc_list)

