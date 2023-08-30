#include "../../includes/my.h"

void my_nn_backprogation(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y, uint32_t size)
{
    my_nn_forward(nn, x, size);
    uint32_t m = y->n;

    MAT_DECLA(dz);

    MAT_DECLA(neg_y);

    my_matrix_multiplybyscalar(y, -1, &neg_y);

    my_matrix_add(&dz, 2, &(nn->activations[size - 1]), &neg_y);

    for (uint32_t i = size - 1; i >= 0; --i) {
        
    }
}