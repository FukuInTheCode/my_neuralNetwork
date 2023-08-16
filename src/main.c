#include "../includes/my.h"

int main(int argc, char* argv[])
{
    uint8_t **layers = {2, 3, 3, 1};
    my_nn_t N = {.layers_size = 0};
    my_nn_create(&N, layers, 4);
    return 0;
}
