#include "../../includes/my.h"

double my_nn_linear(double x)
{
    return x;
}

double my_nn_linear_grad(double x)
{
    if (x == 0) return 1 * 1e-1;
    return x / x * 1e-1;
}
