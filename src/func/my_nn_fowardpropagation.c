#include "../../includes/my.h"

void my_nn_forwardpropagation(my_nn_t *N, my_matrix_t *inputs)
{
    my_matrix_copy(inputs, &(N->activations[0]));
    my_matrix_t tmp = {.m = 0, .n = 0};
    my_matrix_t Z = {.m = 0, .n = 0};
    for (uint32_t i = 1; i < N->layers_size; i++) {
        my_matrix_product(&tmp, 2, &(N->activations[i - 1]), &(N->theta_arr[i - 1]));
        my_matrix_add(&Z, 2, &tmp, &(N->bias_arr[i - 1]));
        if (i < N->layers_size - 1)
            my_matrix_applyfunc(&Z, my_nn_activation_relu, &(N->activations[i]));
        else
            my_matrix_copy(&Z, &(N->activations[i]));
    }
    my_matrix_free(2, &tmp, &Z);
}
