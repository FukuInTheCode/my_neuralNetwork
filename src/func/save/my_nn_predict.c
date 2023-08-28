#include "../../includes/my.h"

void my_nn_predict(my_nn_t *N, my_matrix_t *inputs, my_matrix_t *pred)
{
    my_nn_forwardpropagation(N, inputs);
    my_matrix_copy(&(N->activations[N->layers_size - 1]), pred);
}