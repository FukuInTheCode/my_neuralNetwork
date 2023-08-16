#include "../../includes/my.h"

void my_nn_free(my_nn_t *N)
{
    my_matrix_free_array(&(N->theta_arr), N->layers_size);
    my_matrix_free_array(&(N->bias_arr), N->layers_size);
}
