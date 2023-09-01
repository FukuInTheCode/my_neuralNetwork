#include "../../includes/my.h"

double my_nn_sin(double x)
{
    return sin(x);
}

double my_nn_sin_grad(double x)
{
    return cos(x);
}
