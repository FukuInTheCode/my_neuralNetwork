#include "../../includes/my.h"

void my_nn_forwardpropagation(my_nn_t *N, my_matrix_t *inputs)
{
    my_matrix_product(&(N->activations[1]), 2, inputs, &(N->theta_arr[0]));
}
