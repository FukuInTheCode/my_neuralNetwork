#include "../../includes/my.h"

#define SQRT2 1.4142135624
#define SQRT2PI 2.5066282746

double my_nn_gelu(double x)
{
    return x / 2. * (1 + erf(x / SQRT2));
}

double my_nn_gelu_grad(double x)
{
    return 1. / 2. * (1 + erf(x / SQRT2)) + exp(-1 * pow(x, 2) / 2) * x / SQRT2PI;
}
