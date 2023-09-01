#include "../../includes/my.h"

void my_nn_free(my_nn_t *nn)
{
    my_matrix_free_array(&(nn->gradients_bias), nn->size - 1);
    my_matrix_free_array(&(nn->gradients_theta), nn->size - 1);
    my_matrix_free_array(&(nn->theta_arr), nn->size - 1);
    my_matrix_free_array(&(nn->bias_arr), nn->size - 1);
    my_matrix_free_array(&(nn->activations), nn->size);
}
