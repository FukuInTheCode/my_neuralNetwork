#include "../../includes/my.h"

static double relu_grad(double x) {
    if (x > 0) return 1.0;
    return 0.0;
}

static inline __attribute__((always_inline)) void calc_dz(my_nn_t *N, my_matrix_t *dZ, uint8_t i)
{
    my_matrix_t tmp = {.m = 0, .n = 0};
    my_matrix_t tmp2 = {.m = 0, .n = 0};
    my_matrix_transpose(&(N->theta_arr[i - 1]), &tmp);
    my_matrix_product(&tmp2, 2, dZ, &tmp);
    my_matrix_applyfunc(&(N->activations[i - 1]), relu_grad, &tmp);
    my_matrix_product_elementwise(dZ, 2, &tmp, &tmp2);
    my_matrix_free(2, &tmp, &tmp2);
}

void my_nn_backpropagation(my_nn_t *N, my_matrix_t *inputs, my_matrix_t *Y)
{
    my_nn_forwardpropagation(N, inputs);
    my_matrix_t dZ = {.m = 0, .n = 0};
    my_matrix_t negY = {.m = 0, .n = 0};
    my_matrix_multiplybyscalar(Y, -1, &negY);
    my_matrix_add(&dZ, 2, &(N->activations[N->layers_size - 1]), negY);
    my_matrix_t tmp = {.m = 0, .n = 0};
    my_matrix_t tmp2 = {.m = 0, .n = 0};
    for (uint32_t i = N->layers_size - 1; i > 0; i--) {
        my_matrix_transpose(&(N->activations[i - 1]), &tmp);
        my_matrix_product(&tmp2, 2, &tmp, &dZ);
        my_matrix_multiplybyscalar(&tmp2, 1.0/(double)(Y->m), &(N->gradientsTheta[i - 1]));
        my_matrix_sumcol(&dZ, &tmp);
        my_matrix_multiplybyscalar(&tmp, 1.0/(double)(Y->m), &(N->gradientsBias[i - 1]));
        if (i == 1) continue;
        calc_dz(N, &dZ, i);
    }
    my_matrix_free(4, &dZ, &negY, &tmp, &tmp2);
}