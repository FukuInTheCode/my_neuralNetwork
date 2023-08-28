#include "../includes/my.h"

// AND - gates
double and_train_fea[] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};
double and_train_tar[] = {
    0,
    0,
    0,
    1
};
// OR - gates
double or_train_fea[] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};
double or_train_tar[] = {
    0,
    1,
    1,
    1
};
// XOR - gates
double xor_train_fea[] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};
double xor_train_tar[] = {
    0,
    1,
    1,
    0
};
// UNKNW - gates
double unknw_train_fea[] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};
double unknw_train_tar[] = {
    1,
    0,
    1,
    0
};

static double square(double x)
{
    return x * x;
}

int main(int argc, char* argv[])
{
    srand(69);

    uint32_t n_layers = 1;

    my_matrix_t layers_w[n_layers];
    my_matrix_t layers_b[n_layers];

    MAT_DECLA(tmp);
    MAT_DECLA(w1);
    MAT_DECLA(b);
    MAT_DECLA(features);
    MAT_DECLA(targets);
    MAT_DECLA(predictions);
    MAT_DECLA(diff);

    my_matrix_create(4, 2, 1, &tmp);
    my_matrix_fill_from_array(&tmp, and_train_fea, 8);
    my_matrix_transpose(&tmp, &features);
    MAT_PRINT(features);
    // MAT_PRINT_DIM(features);

    my_matrix_create(4, 1, 1, &tmp);
    my_matrix_fill_from_array(&tmp, and_train_tar, 8);
    my_matrix_transpose(&tmp, &targets);
    MAT_PRINT(targets);

    my_matrix_create(1, 2, 1, &w1);
    my_matrix_randfloat(-1, 1, 1, &w1);
    my_matrix_create(1, 1, 1, &b);
    my_matrix_randfloat(-1, 1, 1, &b);
    // MAT_PRINT(w1);
    // MAT_PRINT_DIM(w1);
    // MAT_PRINT(b);
    // MAT_PRINT_DIM(b);

    my_matrix_product(&tmp, 2, &w1, &features);
    my_matrix_add(&predictions, 2, &tmp, &b);
    MAT_PRINT(predictions);

    my_matrix_multiplybyscalar(&targets, -1, &tmp);

    my_matrix_add(&diff, 2, &tmp, &predictions);

    MAT_PRINT(diff);
    my_matrix_applyfunc(&diff, square, &tmp);

    double sum = my_matrix_sum(&tmp);
    sum /= (double)targets.n;
    printf("sum = %lf\n", sum);



    my_matrix_free(1, &w1);

    return 0;
}
