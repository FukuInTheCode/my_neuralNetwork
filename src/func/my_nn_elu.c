#include "../../includes/my.h"

double my_nn_elu(double *params, double x)
{
    if (x <= 0) return params[0] * (exp(x) - 1);
    return x;
}

double my_nn_elu_grad(double *params, double x)
{
    if (x <= 0) return params[0] * exp(x);
    return 1. * 1e-1;
}
