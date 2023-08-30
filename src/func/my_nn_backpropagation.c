#include "../../includes/my.h"

void my_nn_backprogation(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y, uint32_t size)
{
    my_nn_forward(nn, x, size);
    uint32_t m = y->n;

    MAT_DECLA(dz);

    MAT_DECLA(neg_y);

    my_matrix_multiplybyscalar(y, -1, &neg_y);

    my_matrix_add(&dz, 2, &(nn->activations[size - 1]), &neg_y);

    for (uint32_t i = size - 1; i > 0; --i) {
        MAT_DECLA(at);

        my_matrix_transpose(&(nn->activations[i - 1]), &at);

        MAT_DECLA(dz_dot_at);

        my_matrix_product(&dz_dot_at, 2, &dz, &at);

        my_matrix_multiplybyscalar(&dz_dot_at, 1.0 / m, &(nn->gradients_theta[i - 1]));

        MAT_DECLA(summed_dz);

        my_matrix_sumrow(&dz, &summed_dz);

        my_matrix_multiplybyscalar(&summed_dz, 1.0 / m, &(nn->gradients_bias[i - 1]));

        MAT_DECLA(wt);

        my_matrix_transpose(&(nn->theta_arr[i - 1]), &wt);

        MAT_DECLA(wt_dot_dz);

        my_matrix_product(&wt_dot_dz, 2, &wt, &dz);

        


    }
}