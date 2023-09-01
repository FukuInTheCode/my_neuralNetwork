#include "../../includes/my.h"

double my_nn_softplus(double x)
{
    return log(1 + exp(x));
}

double my_nn_softplus_grad(double x)
{
    return 1. / (1 + exp(-x));
}
