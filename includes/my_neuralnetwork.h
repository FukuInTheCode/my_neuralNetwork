#pragma once
#define MYNEURALNETWORKH

#include "my_matrix.h"

typedef struct my_nn {
    my_matrix *theta_arr;
    my_matrix *bias_arr;
    uint8_t *layers;
    uint8_t layers_size;
} my_nn;
