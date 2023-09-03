#include "../../includes/my.h"

double my_nn_atan(double x)
{
    return atan(x);
}

double my_nn_atan_grad(double x)
{
    return 1. / (pow(x, 2) + 1);
}
