#include "../../includes/my.h"

double my_nn_calcerror_mse(my_nn_t *N, my_matrix_t *inputs, my_matrix_t *Y)
{
    my_nn_forwardpropagation(N, inputs);
    my_matrix_t *predictions = &(N->activations[N->layers_size - 1]);
    my_matrix_print(1, predictions);
}