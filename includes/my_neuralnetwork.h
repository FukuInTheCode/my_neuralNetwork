#pragma once
#define MYNEURALNETWORKH

#include "my_matrix.h"

typedef double (*activation_func)(double);

typedef struct {
    my_matrix_t *theta_arr;
    my_matrix_t *bias_arr;
    my_matrix_t *activations;
    my_matrix_t *gradients_theta;
    my_matrix_t *gradients_bias;
    struct {
        activation_func af;
    } funcs;
} my_nn_t;

void my_nn_create(my_nn_t *nn, uint32_t *dimensions, uint32_t dimensions_size);
void my_nn_forward(my_nn_t *nn, my_matrix_t *x, uint32_t size);
double my_nn_sigmoid(double x);
double my_nn_sig_grad(double x);
void my_nn_backprogation(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y, uint32_t size);
