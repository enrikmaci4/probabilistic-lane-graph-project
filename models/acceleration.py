import functions.graph as graph

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
def linear(ttc: float, dtc: float, A_max=10, T=5, D=10):
    a_ttc = _linear_ttc(ttc=ttc, A_max=A_max, T=T)
    a_dtc = _linear_dtc(dtc=dtc, A_max=A_max, D=D)
    return min(a_ttc, a_dtc)


###############################################################################
# A linear acceleration model used in order to test our framework. The model  #
# Looks something like this:                                                  #
#                                                                             #
# POSITIVE X:               NEGATIVE X:                                       #
#  A_max _|    ____              _______|_ A_max                              #
#         |   /|                        |                                     #
#      0 _|__/_|___ttc      ttc ________|_ 0                                  #
#         | /| |                        |                                     #
# -A_max _|/ T 2T                       |_ -A_max                             #
#         |                             |                                     #
#                                                                             #
# To get the graph shown above we need f(ttc) = m*ttc + c where:              #
#                                                                             #
# => m = A_max/T                                                              #
# => c = -A_max                                                               #
###############################################################################
def _linear_ttc(ttc: float, A_max=10, T=5): 
    if ttc >= 0:
        return _positive_x_linear(ttc=ttc, A_max=A_max, T=T)
    else:
        return _negative_x_linear(ttc=ttc, A_max=A_max, T=T)


def _negative_x_linear(ttc: float, A_max=2, T=5):
    return A_max


def _positive_x_linear(ttc: float, A_max=2, T=5):
    # Get gradient and intercept
    m = A_max/T
    c = -A_max
    return min(m*ttc + c, A_max/4)


###############################################################################
# A linear acceleration model used in order to test our framework. The model  #
# Looks something like this:                                                  #
#                                                                             #
# POSITIVE X:               NEGATIVE X:                                       #
#  A_max _|    ____              _______|_ A_max                              #
#         |   /|                        |                                     #
#      0 _|__/_|___dtc      dtc ________|_ 0                                  #
#         | /| |                        |                                     #
# -A_max _|/ D 2D                       |_ -A_max                             #
#         |                             |                                     #
#                                                                             #
# To get the graph shown above we need f(dtc) = m*dtc + c where:              #
#                                                                             #
# => m = A_max/D                                                              #
# => c = -A_max                                                               #
###############################################################################
def _linear_dtc(dtc: float, A_max=10, D=10): 
    if dtc >= 0:
        return _positive_x_linear(ttc=dtc, A_max=A_max, T=D)
    else:
        return _negative_x_linear(ttc=dtc, A_max=A_max, T=D)

