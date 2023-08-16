#include "../includes/my.h"

int main(int argc, char* argv[])
{
    my_randint(1, 23);
    uint8_t layers[] = {2, 3, 3, 1};
    my_nn_t N = {.layers_size = 0};
    my_nn_create(&N, layers, 4);
    my_matrix_t A = {.m = 0, .n = 0};
    my_matrix_create(1, 3, 1, &A);
    my_matrix_randint(1, 23, 1, &A);
    my_matrix_print(1, &A);
    my_matrix_free(1, &A);
    return 0;
}
