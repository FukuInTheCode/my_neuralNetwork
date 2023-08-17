#pragma once
#define MYNEURALNETWORKH

#include "my_matrix.h"

typedef struct my_nn {
    my_matrix_t *theta_arr;
    my_matrix_t *bias_arr;
    uint8_t *layers;
    uint8_t layers_size;
    my_matrix_t *activations;
} my_nn_t;

void my_nn_create(my_nn_t *N);
void my_nn_free(my_nn_t *N);
void my_nn_create_activation(my_nn_t *N, uint8_t inputs_size);
void my_nn_forwardpropagation(my_nn_t *N, my_matrix_t *inputs);