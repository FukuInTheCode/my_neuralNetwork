#include "../../includes/my.h"

double my_nn_silu(double x)
{
    return x / (1. + exp(-x));
}

double my_nn_silu_grad(double x)
{
    return (1 + exp(-x) + x * exp(-x)) / pow(1 + exp(-x), 2)
}
