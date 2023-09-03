#include "../../includes/my.h"

double my_nn_softsign(double x)
{
    return x / (1 + my_abs(x));
}

double my_nn_softsign_grad(double x)
{
    return 1. / pow(1 + my_abs(x), 2);
}
