#include "../../includes\my.h"

void my_nn_forward(my_nn_t *nn, my_matrix_t *x, uint32_t size)
{
    my_matrix_copy(x, &(nn->activations[0]));
    for (uint32_t i = 1; i < size; ++i) {
        MAT_DECLA(tmp);
        MAT_DECLA(z);
        my_matrix_product(&tmp, 2, &(nn->theta_arr[i - 1]), x);
        MAT_PRINT_DIM(tmp);


        my_matrix_free(2, &tmp, &z);
    }
}
