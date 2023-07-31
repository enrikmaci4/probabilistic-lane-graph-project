import functions.graph as graph

A_MAX = 2.5
A_MIN = -10

###############################################################################
# A linear, multi-variate acceleration model. Run two linear models in        #
# parallel. One model for ttc and one for dtc. Then take the minimum          #
# acceleration between the two. I.e. take the one which brakes the hardest.   #
# This is a "safe" implementation because we're always taking the             #
# acceleration for the worst case scenario.                                   #
#                                                                             #
# Params: ttc - Time to collision value.                                      #
#         dtc - Distance to collision value.                                  #
#         T   - TTC threshold at which we return 0 acceleration (shown        #
#               below).                                                       #
#         D   - DTC threshold at which we return 0 acceleration (shown        #
#               below).                                                       #
#                                                                             #
###############################################################################
def linear(ttc: float, dtc: float, T=5, D=10):
    a_ttc = _linear_ttc(ttc=ttc, T=T)
    a_dtc = _linear_dtc(dtc=dtc, D=D)
    # Return the higher magnitude action
    if abs(a_ttc) > abs(a_dtc):
        return a_ttc
    else:
        return a_dtc


###############################################################################
# A linear acceleration model used in order to test our framework. The model  #
# Looks something like this:                                                  #
#                                                                             #
# POSITIVE X:               NEGATIVE X:                                       #
#  A_max _|    ____         - Same as positive X but reflected on the         #
#         |   /|              vertical and horizontal axis.                   #
#      0 _|__/_|___ttc                                                        #
#         | /| |                                                              #
# -A_min _|/ T                                                                #
#         |                                                                   #
#                                                                             #
# To get the graph shown above we need f(ttc) = m*ttc + c where:              #
#                                                                             #
# => m = (0-A_min)/T                                                          #
# => c = A_min                                                                #
###############################################################################
def _linear_ttc(ttc: float, T=5): 
    if ttc >= 0:
        return _positive_x_linear(ttc=ttc, T=T)
    else:
        return _negative_x_linear(ttc=ttc, T=T)


def _negative_x_linear(ttc: float, T=5):
    return -_positive_x_linear(ttc=-ttc, T=T)


def _positive_x_linear(ttc: float, T=5):
    # Get gradient and intercept
    m = -A_MIN/T
    c = A_MIN
    return min(m*ttc + c, A_MAX)


###############################################################################
# A linear acceleration model used in order to test our framework. The model  #
# Looks something like this:                                                  #
#                                                                             #
# POSITIVE X:               NEGATIVE X:                                       #
#  A_max _|    ____         - Same as positive X but reflected on the         #
#         |   /|              vertical and horizontal axis.                   #
#      0 _|__/_|___dtc                                                        #
#         | /| |                                                              #
# -A_min _|/ D                                                                #
#         |                                                                   #
#                                                                             #
# To get the graph shown above we need f(dtc) = m*dtc + c where:              #
#                                                                             #
# => m = (0-A_min)/D                                                          #
# => c = A_min                                                                #
###############################################################################
def _linear_dtc(dtc: float, D=10): 
    if dtc >= 0:
        return _positive_x_linear(ttc=dtc, T=D)
    else:
        return _negative_x_linear(ttc=dtc, T=D)

