#include "../../includes/my.h"

double my_nn_elu(my_nn_t *nn, double x)
{
    if (x <= 0) return nn->funcs.params[0] * (exp(x) - 1);
    return x;
}

double my_nn_elu_grad(my_nn_t *nn, double x)
{
    if (x <= 0) return nn->funcs.params[0] * exp(x);
    return 1.;
}
