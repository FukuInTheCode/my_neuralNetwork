#include "../../includes/my.h"

double my_nn_calc_error(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y)
{

    MAT_DECLA(p);

    my_nn_predict(nn, x, &p);

    MAT_DECLA(neg_y);

    my_matrix_multiplybyscalar(y, -1, &neg_y);

    MAT_DECLA(diff);

    my_matrix_add(&diff, 2, &p, &neg_y);

    double res = my_matrix_sum(&diff);

    res /= y->n;

    return res;
}
