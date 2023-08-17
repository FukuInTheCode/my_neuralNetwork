#include "../includes/my.h"

int main(int argc, char* argv[])
{
    my_matrix_t features = {.m = 0, .n = 0};
    my_matrix_create(10, 2, 1, &features);
    my_matrix_randint(-1, 1, 1, &features);
    my_matrix_print(1, &features);
    uint8_t layers[] = {2, 3, 1};
    my_nn_t N = {.layers_size = 3, .layers = layers};
    my_nn_create(&N);
    my_nn_create_activation(&N, features.m);

    my_matrix_print_array(&(N.theta_arr), N.layers_size - 1);
    my_matrix_print_array(&(N.bias_arr), N.layers_size - 1);
    my_nn_forwardpropagation(&N, &features);
    my_matrix_print_array(&(N.activations), N.layers_size);

    my_nn_free(&N);
    my_matrix_free(1, &features);
    return 0;
}
