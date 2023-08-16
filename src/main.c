#include "../includes/my.h"

int main(int argc, char* argv[])
{
    my_matrix_t features = {.m = 0, .n = 0};
    my_matrix_t *acti;
    my_matrix_create(10, 2, 1, &features);
    my_matrix_print(1, &features);
    uint8_t layers[] = {2, 2, 1};
    my_nn_t N = {.layers_size = 0};
    my_nn_create(&N, layers, 3);
    my_nn_forwardpropagation(&N, &features, &acti);
    my_matrix_print_array(&acti, N.layers_size);
    my_nn_free(&N);
    my_matrix_free(1, &features);
    return 0;
}
