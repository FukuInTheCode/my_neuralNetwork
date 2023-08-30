#include "../../includes/my.h"

double my_nn_relu_grad(double x)
{
    if (x > 0) return 1;
    return 0;
}