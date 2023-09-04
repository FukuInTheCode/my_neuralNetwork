#include "../../includes/my.h"

void my_nn_print(my_nn_t *nn)
{
    printf("-- %s's Weights --\n", nn->name);
    my_matrix_print_array(&(nn->theta_arr), nn->size - 1);
    printf("-- %s's Bias --\n", nn->name);
    my_matrix_print_array(&(nn->bias_arr), nn->size - 1);
    printf("---  ---  ---\n");
}
