#include "../../includes/my.h"

void my_nn_print(my_nn_t *N)
{
    my_matrix_print_array(&(N->theta_arr), N->layers_size - 1);
    my_matrix_print_array(&(N->bias_arr), N->layers_size - 1);
}
