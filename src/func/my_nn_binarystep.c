#include "../../includes/my.h"

double my_nn_binarystep(double x)
{
    if (x <= 0) return 0;
    return 1;
}

double my_nn_binarystep_grad(double x)
{
    return 0;
}
