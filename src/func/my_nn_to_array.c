#include "../../includes/my.h"

void my_nn_to_array(my_nn_t *nn, double **res)
{
    uint32_t k = 0;
    for (uint32_t i = 0; i < nn->size - 1; ++i) {
        double *arr = malloc(sizeof(double) * nn->theta_arr[i].m * nn->theta_arr[i].n);
        my_matrix_to_array(&(nn->theta_arr[i]), &arr);
        for (uint32_t j = 0; j < nn->theta_arr[i].m * nn->theta_arr[i].n; ++j)
            (*res)[k + j] = arr[j];
        free(arr);
        k += nn->theta_arr[i].m * nn->theta_arr[i].n;
        arr = malloc(sizeof(double) * nn->bias_arr[i].m * nn->bias_arr[i].n);
        my_matrix_to_array(&(nn->bias_arr[i]), &arr);
        for (uint32_t j = 0; j < nn->bias_arr[i].m * nn->bias_arr[i].n; ++j)
            (*res)[k + j] = arr[j];
        k += nn->bias_arr[i].m * nn->bias_arr[i].n;
        free(arr);
    }
}
