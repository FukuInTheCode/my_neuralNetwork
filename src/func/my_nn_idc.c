#include "../../includes/my.h"

double my_nn_idc(double x)
{
    return (sqrt(pow(x, 2) + 1.) - 1.) / 2. + x;
}

double my_nn_idc_grad(double x)
{
    return x / (2. * sqrt(pow(x, 2) + 1.)) + 1.;
}
