#include "../../includes/my.h"

static double __square(double x) {
    return x * x;
}

double my_nn_calcerror_mse(my_nn_t *N, my_matrix_t *inputs, my_matrix_t *Y)
{
    my_nn_forwardpropagation(N, inputs);
    my_matrix_t *predictions = &(N->activations[N->layers_size - 1]);
    my_matrix_t negY = {.m = 0, .n = 0};
    my_matrix_multiplybyscalar(Y, -1, &negY);
    my_matrix_t diff = {.m = 0, .n = 0};
    my_matrix_add(&diff, 2, &negY, predictions);
    my_matrix_t squared_diff = {.m = 0, .n = 0};
    my_matrix_applyfunc(&diff, __square, &squared_diff);
    double sum = my_matrix_sum(&squared_diff);
    my_matrix_free(3, &negY, &diff, &squared_diff);
    return sum / Y->m;
}