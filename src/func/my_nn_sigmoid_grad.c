#include "../../includes/my.h"

double my_nn_activation_sigmoid_grad(double x)
{
    return my_nn_activation_sigmoid(x) * my_nn_activation_sigmoid(1 - x);
}
