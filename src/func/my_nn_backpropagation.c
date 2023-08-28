#include "../../includes/my.h"

void  my_nn_backpropagation(my_nn_t *N, my_matrix_t *inputs, my_matrix_t *Y)
{
    my_nn_forwardpropagation(N, inputs);
    my_matrix_t neg_y = {.m = 0, .n = 0, .name = "-Y"};
    my_matrix_multiplybyscalar(Y, -1, &neg_y);
    my_matrix_t diff = {.m = 0, .n = 0, .name = "diff"};
    my_matrix_add(&diff, 2, &(N->activations[N->layers_size - 1]), &neg_y);
    my_matrix_t dz = {.m = 0, .n = 0, .name = "dZ"};
    my_matrix_t dsigz = {.m = 0, .n = 0, .name = "delta sig z"};
    if (N->apply_all) {
        my_matrix_applyfunc(&(N->z[N->layers_size - 2]), my_nn_activation_sigmoid_grad, &dsigz);
        my_matrix_product_elementwise(&dz, 2, &diff, &dsigz);
    } else
        my_matrix_copy(&diff, &dz);
    for (uint32_t i = 1; i < N->layers_size; ++i) {
        // MAT_PRINT(dz);
        my_matrix_t a_trsp = {.m = 0, .n = 0, .name = "AT"};
        my_matrix_transpose(&(N->activations[N->layers_size - 1 - i]), &a_trsp);
        // MAT_PRINT(a_trsp);
        my_matrix_t not_mean = {.m = 0, .n = 0, .name = "not_mean"};
        my_matrix_product(&not_mean, 2, &a_trsp, &dz);
        // MAT_PRINT(not_mean);
        my_matrix_multiplybyscalar(&not_mean, 1.0 / (double)Y->m, &(N->gradientsTheta[N->layers_size - 1 - i]));
        // MAT_PRINT((N->gradientsTheta[N->layers_size - 1 - i]));
        my_matrix_t sum_axe_m = {.m = 0, .n = 0, .name = "sum axe m"};
        my_matrix_sumcol(&dz, &sum_axe_m);
        // MAT_PRINT(sum_axe_m);
        my_matrix_multiplybyscalar(&sum_axe_m, 1.0 / (double)(Y->m), &(N->gradientsBias[N->layers_size - 1 - i]));
        // MAT_PRINT((N->gradientsBias[N->layers_size - 1 - i]));
        my_matrix_free(3, &a_trsp, &not_mean, &sum_axe_m);
        if (i >= N->layers_size - 1) continue;
        my_matrix_t theta_trsp = {.m = 0, .n = 0, .name = "ThetaT"};
        my_matrix_transpose(&(N->theta_arr[N->layers_size - 1 - i]), &theta_trsp);
        // MAT_PRINT(theta_trsp);
        my_matrix_t p_with_theta = {.m = 0, .n = 0, .name = "produ theta"};
        my_matrix_product(&p_with_theta, 2, &dz, &theta_trsp);
        // MAT_PRINT(p_with_theta);
        // MAT_PRINT((N->z[N->layers_size - 2 - i]));
        my_matrix_applyfunc(&(N->z[N->layers_size - 2 - i]), my_nn_activation_sigmoid_grad, &dsigz);
        // MAT_PRINT(dsigz);
        my_matrix_product_elementwise(&dz, 2, &p_with_theta, &dsigz);
        my_matrix_free(2, &theta_trsp, &p_with_theta);
    }
    my_matrix_free(4, &neg_y, &diff, &dsigz, &dz);
}
