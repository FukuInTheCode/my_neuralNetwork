#include "../../includes\my.h"

static apply_func_with_params(my_nn_t *nn, my_matrix_t *A, my_matrix_t *result)
{

    my_matrix_copy(A, result);

    for (uint32_t i = 0; i < result->m; i++) {
        for (uint32_t j = 0; j < result->n; j++) {
            my_matrix_set(result, j, i, nn->funcs.af_p(nn->funcs.params, \
                                                        result->arr[i][j]));
        }
    }
}

void my_nn_forward(my_nn_t *nn, my_matrix_t *x)
{
    my_matrix_copy(x, &(nn->activations[0]));

    for (uint32_t i = 1; i < nn->size; ++i) {
        MAT_DECLA(tmp);
        MAT_DECLA(z);

        my_matrix_product(&tmp, 2, &(nn->theta_arr[i - 1]), \
                                    &(nn->activations[i - 1]));

        my_matrix_add(&z, 2, &tmp, &(nn->bias_arr[i - 1]));

        if (nn->acti_type == base_type)
            my_matrix_applyfunc(&z, nn->funcs.af, &(nn->activations[i]));
        else
            apply_func_with_params(nn->funcs.params, &z,\
                                        &(nn->activations[i]));

        my_matrix_free(2, &tmp, &z);
    }
}
