#include "../../includes/my.h"

void my_nn_train(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y, my_params_t *hp)
{
    for (uint32_t j = 0; j < hp->epoch; ++j) {
        for (uint32_t i = 0; i < nn->size - 1; ++i) {
            MAT_DECLA(tmp);
            my_matrix_multiplybyscalar(&(nn->gradients_theta[i]), -1 * hp->alpha, &tmp);
            MAT_DECLA(cpy);
            my_matrix_copy(&(nn->theta_arr[i]), &cpy);
            my_matrix_add(&(nn->theta_arr[i]), 2, &cpy, &tmp);

            my_matrix_multiplybyscalar(&(nn->gradients_bias[i]), -1 * hp->alpha, &tmp);
            my_matrix_copy(&(nn->bias_arr[i]), &cpy);
            my_matrix_add(&(nn->bias_arr[i]), 2, &cpy, &tmp);
        }
    }
}
