#include "../../includes/my.h"

double my_nn_gaussian(double x)
{
    return exp(-1. * pow(x, 2));
}

double my_nn_gaussian_grad(double x)
{
    return -2. * x * exp(-1. * pow(x, 2));
}
