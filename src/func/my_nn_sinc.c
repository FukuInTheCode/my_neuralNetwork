#include "../../includes/my.h"

double my_nn_sinc(double x)
{
    if (x == 0) return 1;
    return sin(x) / x;
}

double my_nn_sinc_grad(double x)
{
    if (x == 0) return 0;
    return cos(x) / x - sin(x) / pow(x, 2);
}
