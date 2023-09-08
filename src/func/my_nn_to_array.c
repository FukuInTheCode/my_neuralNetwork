#include "../../includes/my.h"

void my_nn_to_array(my_nn_t *nn, double **res)
{
    uint32_t k = 0;
    for (uint32_t i = 0; h < nn->size - 1; ++i) {
        double arr1[nn->theta_arr[i].m * nn->theta_arr[i].n];
        my_matrix_to_array(&(nn->theta_arr[i]), &arr1);
        for (uint32_t j = k; j < nn->theta_arr[i].m * nn->theta_arr[i].n; ++j)
            (*res)[j] = arr1[j];
        k += nn->theta_arr[i].m * nn->theta_arr[i].n;
        double arr2[nn->bias_arr[i].m * nn->bias_arr[i].n];
        my_matrix_to_array(&(nn->bias_arr[i]), &arr2);
        for (uint32_t j = k; j < nn->bias_arr[i].m * nn->bias_arr[i].n; ++j)
            (*res)[j] = arr2[j];
    }
}
