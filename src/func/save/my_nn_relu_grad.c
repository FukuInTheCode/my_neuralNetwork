#include "../../includes/my.h"

double my_nn_activation_relu_grad(double x)
{
    if (x > 0) return 1.0;
    return 0.0;
}
