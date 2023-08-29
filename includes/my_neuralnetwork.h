#pragma once
#define MYNEURALNETWORKH

#include "my_matrix.h"

typedef double (*activation_func)(double);

typedef struct {
    my_matrix_t *theta_arr;
    my_matrix_t *bias_arr;
    my_matrix_t *activations;
    struct {
        activation_func af;
    } funcs;
} my_nn_t;

void my_nn_create(my_nn_t *nn, uint32_t *dimensions, uint32_t dimensions_size);
void my_nn_forward(my_nn_t *nn, my_matrix_t *x, uint32_t size);
