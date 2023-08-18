#include "../../includes/my.h"

void my_nn_train(my_nn_t *N, my_matrix_t *inputs, \
                my_matrix_t *Y, my_params_t *hyper_params)
{
    my_matrix_t copy = {.m = 0, .n = 0};
    my_matrix_t tmp = {.m = 0, .n = 0};
    uint32_t i = 0;
    for ( ; i < hyper_params->iterations; i++) {
        my_nn_backpropagation(N, inputs, Y);
        if (my_nn_calcerror_mse(N, inputs, Y) <= hyper_params->threshold) break;
        for (uint32_t j = 0; j < N->layers_size - 1; j++) {
            my_matrix_copy(&(N->gradientsTheta[j]), &copy);
            my_matrix_multiplybyscalar(&copy, -(hyper_params->alpha), &tmp);
            my_matrix_copy(&(N->theta_arr[j]), &copy);
            my_matrix_add(&(N->theta_arr[j]), 2, &copy, &tmp);
            my_matrix_copy(&(N->gradientsBias[j]), &copy);
            my_matrix_multiplybyscalar(&copy, -(hyper_params->alpha), &tmp);
            my_matrix_copy(&(N->bias_arr[j]), &copy);
            my_matrix_add(&(N->bias_arr[j]), 2, &copy, &tmp);
        }
    }
    printf("took %u iterations to train!\n", i);
    my_matrix_free(2, &copy, &tmp);
}
