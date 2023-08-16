#include "../../includes/my.h"

void my_nn_forwardpropagation(my_nn_t *N, my_matrix_t *inputs,\
                                my_matrix_t **activations)
{
    uint8_t i;

    *activations = malloc((N->layers_size - 1)\
                                * sizeof(my_matrix_t));
    if (*activations == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    (*activations)[0] = *inputs;
    my_matrix_t Z = {.m = 0, .n = 0};
    for (i = 0; i < N->layers_size; i++) {
        my_matrix_product(&Z, 2, &(*activations)[i - 1],\
                                    &(N->theta_arr[i - 1]));
        my_matrix_add(&((*activations)[i]), 2, &Z, &(N->bias_arr[i - 1]));
    }
    my_matrix_free(1, &Z);
}
