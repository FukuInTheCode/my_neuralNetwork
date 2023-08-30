#include "../../includes/my.h"

void my_nn_predict(my_nn_t *nn, my_matrix_t *x, my_matrix_t *res)
{
    my_nn_forward(nn, x);
    my_matrix_copy(&(nn->activations[nn->size - 1]), res);
}
