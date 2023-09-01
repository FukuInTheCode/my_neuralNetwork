#include "../../includes/my.h"

double my_nn_relu_grad(double x)
{
    return tanh(x);
}
