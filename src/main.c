#include "../includes/my.h"

int main(int argc, char* argv[])
{
    srand(time(NULL));
    my_matrix_t features = {.m = 0, .n = 0};
    my_matrix_t needed = {.m = 0, .n = 0};
    my_matrix_t targets = {.m = 0, .n = 0};
    my_matrix_create(10, 2, 1, &features);
    my_matrix_create(2, 1, 1, &needed);
    my_matrix_randint(-1, 1, 1, &features);
    my_matrix_randint(1, 1, 1, &needed);
    my_matrix_product(&targets, 2, &features, &needed);
    my_matrix_print(2, &features, &targets);
    uint8_t layers[] = {2, 3, 1};
    my_nn_t N = {.layers_size = 3, .layers = layers};
    my_nn_create(&N);
    my_nn_create_activation(&N, features.m);

    my_matrix_print_array(&(N.theta_arr), N.layers_size - 1);
    my_matrix_print_array(&(N.bias_arr), N.layers_size - 1);
    my_nn_calcerror_mse(&N, &features, &targets);

    my_nn_free(&N);
    my_matrix_free(1, &features);
    return 0;
}
