#include "../../includes/my.h"

double my_nn_selu(double *params, double x)
{
    if (x <= 0) return params[0] * params[1] * (exp(x) - 1);
    return params[0] * x;
}

double my_nn_selu_grad(double *params, double x)
{
    if (x <= 0) return params[0] * params[1] * exp(x);
    return params[0];
}
