#include "../../includes/my.h"

void my_nn_from_array(my_nn_t *nn, double *arr)
{
    uint32_t k = 0;
    for (uint32_t i = 0; i < nn->size - 1; ++i) {
        uint32_t size = nn->theta_arr[i].m * nn->theta_arr[i].n;
        double *tmp_arr = malloc(sizeof(double) * size);
        for (uint32_t j = 0; j < size; ++j)
            tmp_arr[j] = arr[j + k];
        my_matrix_fill_from_array(&(nn->theta_arr[i]), tmp_arr, size);
        free(tmp_arr);
        k += size;
        size = nn->bias_arr[i].m * nn->bias_arr[i].n;
        tmp_arr = malloc(sizeof(double) * size);
        for (uint32_t j = 0; j < size; ++j)
            tmp_arr[j] = arr[j + k];
        my_matrix_fill_from_array(&(nn->bias_arr[i]), tmp_arr, size);
        free(tmp_arr);

    }
}
