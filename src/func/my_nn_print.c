#include "../../includes/my.h"

void my_nn_print(my_nn_t *N)
{
    for (uint8_t i = 0; i < 10; ++i)
        putchar('-');
    printf(" %s's Weights ",N->name);
    for (uint8_t i = 0; i < 10; ++i)
        putchar('-');
    putchar('\n');
    my_matrix_print_array(&(N->theta_arr), N->layers_size - 1);

    for (uint8_t i = 0; i < 10; ++i)
        putchar('-');
    printf(" %s's Bias ",N->name);
    for (uint8_t i = 0; i < 13; ++i)
        putchar('-');
    putchar('\n');
    my_matrix_print_array(&(N->bias_arr), N->layers_size - 1);

    for (uint8_t i = 0; i < 32 + strlen(N->name); ++i)
        putchar('-');
}
