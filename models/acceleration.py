###############################################################################
# A linear acceleration model used in order to test our framework. The model  #
# Looks something like this:                                                  #
#                                                                             #
#  A_max _|    ____                                                           #
#         |   /|                                                              #
#      0 _|__/_|___ttc                                                        #
#         | /| |                                                              #
# -A_max _|/ T 2T                                                             #
#         |                                                                   #
#                                                                             #
# To get the graph shown above we need f(ttc) = m*ttc + c where:              #
#                                                                             #
# => m = A_max/T                                                              #
# => c = -A_max                                                               #
###############################################################################
def linear(ttc: float, A_max=2, T=1):
    # Check the input is > 0
    assert ttc > 0
    # Get gradient and intercept
    m = A_max/T
    c = -A_max
    return min(m*ttc + c, A_max)