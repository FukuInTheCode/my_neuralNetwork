#include "../../includes/my.h"

void my_nn_forwardpropagation(my_nn_t *N, my_matrix_t *inputs)
{
    my_matrix_copy(inputs, &(N->activations[0]));
    my_matrix_t tmp = {.m = 0, .n = 0};
    for (uint32_t i = 1; i < N->layers_size; i++) {
        my_matrix_product(&tmp, 2, \
                    &(N->activations[i - 1]), &(N->theta_arr[i - 1]));
        my_matrix_add(&(N->z[i - 1]), 2, &tmp, &(N->bias_arr[i - 1]));
        if (!(N->apply_all)) {
            if (i < N->layers_size - 1)
                my_matrix_applyfunc(&(N->z[i - 1]), my_nn_activation_sigmoid, \
                                    &(N->activations[i]));
            else
                my_matrix_copy(&(N->z[i - 1]), &(N->activations[i]));
        } else
            my_matrix_applyfunc(&(N->z[i - 1]), my_nn_activation_sigmoid, \
                                    &(N->activations[i]));
    }
    my_matrix_free(1, &tmp);
}
