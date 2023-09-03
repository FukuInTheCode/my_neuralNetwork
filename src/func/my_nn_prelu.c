#include "../../includes/my.h"

double my_nn_prelu(double *params, double x)
{
    if (x < 0) return params[0] * x;
    return x;
}

double my_nn_prelu_grad(double *params, double x)
{
    if (x < 0) return params[0] * 1e-1;
    return 1. * 1e-1;
}
