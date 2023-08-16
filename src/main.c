#include "../includes/my.h"

int main(int argc, char* argv[])
{
    my_randint(1, 23);
    uint8_t layers[] = {2, 3, 3, 1};
    my_nn_t N = {.layers_size = 0};
    my_nn_create(&N, layers, 4);
    my_nn_free(&N);
    return 0;
}
