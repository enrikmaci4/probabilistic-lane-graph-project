import functions.graph as graph

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
def linear(ttc: float, A_max=10, T=5): 
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
    return min(m*ttc + c, A_max)