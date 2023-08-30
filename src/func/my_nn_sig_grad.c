#include "../../includes/my.h"

double my_nn_sig_grad(double x)
{
    return exp(-x) / pow(1 + exp(-x), 2);
}
