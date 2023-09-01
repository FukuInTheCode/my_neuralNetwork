#include "../../includes/my.h"

double my_nn_sigmoid(double x)
{
    return 1.0 / (1 + exp(-x));
}

double my_nn_sigmoid_grad(double x)
{
    return exp(-x) / pow(1 + exp(-x), 2);
}
