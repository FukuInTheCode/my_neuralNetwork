#include "../../includes/my.h"

double my_nn_linear_grad(double x)
{
    if (x == 0) return 1;
    return x / x;
}