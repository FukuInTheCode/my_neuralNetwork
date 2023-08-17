#include "../../includes/my.h"

void my_nn_backpropagation(my_nn_t *N, my_matrix_t *inputs, my_matrix_t *Y)
{
    my_nn_forwardpropagation(N, inputs);
    my_matrix_t dZ = {.m = 0, .n = 0};
    my_matrix_t negY = {.m = 0, .n = 0};
}