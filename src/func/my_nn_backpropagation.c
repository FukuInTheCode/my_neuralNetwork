#include "../../includes/my.h"

static void apply_func_with_params(my_nn_t *nn, my_matrix_t *A,\
                                                my_matrix_t *result)
{
    my_matrix_copy(A, result);

    for (uint32_t i = 0; i < result->m; i++) {
        for (uint32_t j = 0; j < result->n; j++) {
            my_matrix_set(result, j, i, nn->funcs.grad_af_p(nn->funcs.params, \
                                                        result->arr[i][j]));
        }
    }
}

static void calc_dz(my_nn_t *nn, my_matrix_t *dz, uint32_t i)
{
        MAT_DECLA(wt);
        my_matrix_transpose(&(nn->theta_arr[i - 1]), &wt);
        MAT_DECLA(wt_dot_dz);
        my_matrix_product(&wt_dot_dz, 2, &wt, dz);
        MAT_DECLA(tmp);
        MAT_DECLA(z);
        my_matrix_product(&tmp, 2, &(nn->theta_arr[i - 2]),\
                                    &(nn->activations[i - 2]));
        my_matrix_add(&z, 2, &tmp, &(nn->bias_arr[i - 2]));
        MAT_DECLA(grad_a);
        if (nn->acti_type == base_type)
            my_matrix_applyfunc(&z, nn->funcs.grad_af, &grad_a);
        else
            apply_func_with_params(nn, &z, &grad_a);

        my_matrix_product_elementwise(dz, 2, &wt_dot_dz, &grad_a);
        my_matrix_free(5, &wt, &wt_dot_dz, &tmp, &z, &grad_a);
}

void my_nn_backprogation(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y)
{
    my_nn_forward(nn, x);
    MAT_DECLA(dz);
    MAT_DECLA(neg_y);
    my_matrix_multiplybyscalar(y, -1, &neg_y);
    my_matrix_add(&dz, 2, &(nn->activations[nn->size - 1]), &neg_y);
    for (uint32_t i = nn->size - 1; i > 0; --i) {
        MAT_DECLA(at);
        my_matrix_transpose(&(nn->activations[i - 1]), &at);
        my_matrix_product(&(nn->gradients_theta[i - 1]), 2, &dz, &at);
        my_matrix_multiplybyscalar_2(&(nn->gradients_theta[i - 1]),\
                                                            1.0 / y->n);
        my_matrix_sumrow(&dz, &(nn->gradients_bias[i - 1]));
        my_matrix_multiplybyscalar_2(&(nn->gradients_bias[i - 1]), 1.0 / y->n);
        my_matrix_free(1, &at);
        if (i == 1) break;
        calc_dz(nn, &dz, i);
    }
    my_matrix_free(2, &dz, &neg_y);
}