#include "../../includes/my.h"

void my_nn_copy(my_nn_t *nn, my_nn_t *copy)
{
    copy->size = nn->size;
    copy->dims = nn->dims;
    my_matrix_copy_array(&(nn->theta_arr), &(copy->theta_arr), nn->size - 1, "W");
    my_matrix_copy_array(&(nn->bias_arr), &(copy->bias_arr), nn->size - 1, "b");
}
