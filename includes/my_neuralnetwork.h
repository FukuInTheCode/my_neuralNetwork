#pragma once
#define MYNEURALNETWORKH

#include "my_matrix.h"

typedef struct my_nn {
    my_matrix_t *theta_arr;
    my_matrix_t *bias_arr;
    uint32_t *layers;
    uint32_t layers_size;
    my_matrix_t *activations;
    my_matrix_t *gradientsTheta;
    my_matrix_t *gradientsBias;
} my_nn_t;

typedef struct my_params {
    uint32_t iterations;
    double alpha;
    double threshold;
} my_params_t;

void my_nn_create(my_nn_t *N);
void my_nn_free(my_nn_t *N);
void my_nn_create_activation(my_nn_t *N, uint8_t inputs_size);
void my_nn_forwardpropagation(my_nn_t *N, my_matrix_t *inputs);
double my_nn_activation_relu(double x);
double my_nn_calcerror_mse(my_nn_t *N, my_matrix_t *inputs, my_matrix_t *Y);
void my_nn_create_gradients(my_nn_t *N);
void my_nn_backpropagation(my_nn_t *N, my_matrix_t *inputs, my_matrix_t *Y);
void my_nn_train(my_nn_t *N, my_matrix_t *inputs, my_matrix_t *Y, my_params_t *hyper_params);