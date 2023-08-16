#include "../../includes/my.h"

void my_nn_create(my_nn_t *N, uint8_t layers[], \
    const uint8_t layers_size)
{
    unsigned int i;

    uint8_t m = layers[0];

    N->theta_arr = malloc((layers_size - 1) * sizeof(my_matrix *));
    if (N->theta_arr == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    for (i = 1; i < layers_size; i++) {
        uint8_t n = layers[i];
        my_matrix_create(m)
        m = n;
    }
}
