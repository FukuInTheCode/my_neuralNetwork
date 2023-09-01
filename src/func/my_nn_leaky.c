#include "../../includes/my.h"

double my_nn_leaky(double x)
{
    if (x <= 0) return 0.01 * x;
    return x;
}

double my_nn_leaky_grad(double x)
{
    if (x <= 0) return 0.01;
    return 1.;
}
