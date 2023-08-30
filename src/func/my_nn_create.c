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

void my_nn_create(my_nn_t *nn, uint32_t *dimensions)
{
    alloc_arr("W", &(nn->theta_arr), nn->size - 1);
    alloc_arr("b", &(nn->bias_arr), nn->size - 1);
    alloc_arr("A", &(nn->activations), nn->size);
    alloc_arr("db", &(nn->gradients_bias), nn->size - 1);
    alloc_arr("dW", &(nn->gradients_theta), nn->size - 1);
    for (uint32_t i = 0; i < nn->size - 1; ++i) {
        my_matrix_create(dimensions[i + 1], dimensions[i], 1, &(nn->theta_arr[i]));
        my_matrix_randfloat(0, 1, 1, &(nn->theta_arr[i]));
        my_matrix_create(dimensions[i + 1], 1, 1, &(nn->bias_arr[i]));
        my_matrix_randfloat(0, 1, 1, &(nn->bias_arr[i]));
    }
}