#include "../../includes/my.h"

static double square(double x)
{
    return x * x;
}

double my_nn_calc_error(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y)
{

    MAT_DECLA(p);

    my_nn_predict(nn, x, &p);

    MAT_DECLA(neg_y);

    my_matrix_multiplybyscalar(y, -1, &neg_y);

    MAT_DECLA(diff);

    my_matrix_add(&diff, 2, &p, &neg_y);

    MAT_DECLA(diff_squared);

    my_matrix_applyfunc(&diff, square, &diff_squared);

    double res = my_matrix_sum(&diff_squared);

    res /= y->n;

    return res;
}
