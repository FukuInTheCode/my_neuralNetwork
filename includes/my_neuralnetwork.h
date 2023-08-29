#pragma once
#define MYNEURALNETWORKH

#include "my_matrix.h"

typedef struct {
    my_matrix_t *theta_arr;
    my_matrix_t *bias_arr;
} my_nn_t;

void my_nn_create(my_nn_t *nn, uint32_t *dimensions, uint32_t dimensions_size);
