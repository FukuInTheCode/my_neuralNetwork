#include "../../includes/my.h"

double my_nn_softexp(double *params, double x)
{
    if (params[0] < 0)
        return -1 * log(1 - params[0] * (x + params[0])) / params[0];
    else if (params[0] > 0)
        return (exp(params[0] * x) - 1) / params[0];
    return x;
}

double my_nn_softexp_grad(double *params, double x)
{
    if (params[0] < 0)
        return 1. / (1. - params[0] * (params[0] + x));
    return exp(params[0] * x);
}
