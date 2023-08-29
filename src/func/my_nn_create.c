#include "../../includes/my.h"

static void alloc_arr(char *name, my_matrix_t **arr, uint32_t size)
{
    *arr = malloc(size * sizeof(my_matrix_t));
    check_alloc(*arr);
    for (uint32_t i = 0; i < size; ++i) {
        (*arr)[i].name = init_str(name, i + 1);
        (*arr)[i].m = 0;
        (*arr)[i].n = 0;
    }
}

void my_nn_create(my_nn_t *nn, uint32_t *dimensions, uint32_t dimensions_size)
{
    alloc_arr("W", &(nn->theta_arr), dimensions_size - 1);
    alloc_arr("b", &(nn->bias_arr), dimensions_size - 1);
    alloc_arr("A", &(nn->activations), dimensions_size);
    for (uint32_t i = 0; i < dimensions_size - 1; ++i) {
        my_matrix_create(dimensions[i + 1], dimensions[i], 1, &(nn->theta_arr[i]));
        my_matrix_randfloat(0, 1, 1, &(nn->theta_arr[i]));
        MAT_PRINT_DIM(nn->theta_arr[i]);
        my_matrix_create(dimensions[i + 1], 1, 1, &(nn->bias_arr[i]));
        my_matrix_randfloat(0, 1, 1, &(nn->bias_arr[i]));
        MAT_PRINT_DIM(nn->bias_arr[i]);
    }
}