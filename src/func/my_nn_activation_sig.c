#include "../../includes/my.h"

double my_nn_activation_sigmoid(double x)
{
    return 1.0 / (1 + exp(-x));
}
