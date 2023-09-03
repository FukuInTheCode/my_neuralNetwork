#include "../../includes/my.h"

double my_nn_relu(double x)
{
    if (x > 0) return x;
    return 0;
}

double my_nn_relu_grad(double x)
{
    if (x > 0) return 1 * 1e-1;
    return 0;
}
