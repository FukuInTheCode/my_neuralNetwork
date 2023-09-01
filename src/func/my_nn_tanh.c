#include "../../includes/my.h"

double my_nn_tanh(double x)
{
    return tanh(x);
}

double my_nn_tanh_grad(double x)
{
    return 1. - pow(tanh(x), 2);
}
