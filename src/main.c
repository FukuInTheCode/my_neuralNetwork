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

static double fun(double x)
{
    return x * 2;
}

int main(int argc, char* argv[])
{
    srand(69);

    MAT_DECLA(features);
    MAT_DECLA(targets);

    my_matrix_create(1, 10, 1, &features);
    my_matrix_randint(0, 100, 1, &features);
    my_matrix_applyfunc(&features, fun, &targets);

    MAT_PRINT(features);
    MAT_PRINT(targets);

    my_nn_t nn;

    nn.funcs.af = my_nn_sigmoid;
    nn.funcs.grad_af = my_nn_sig_grad;
    nn.size = 2;

    uint32_t dims[] = {1, 1};

    my_nn_create(&nn, dims);

    my_params_t hparams = {
        .alpha = 1e-2,
        .epoch = 100,
        .threshold = 1e-10
    };

    MAT_DECLA(predictions);

    my_nn_predict(&nn, &features, &predictions);

    printf("error: %lf\n", my_nn_calc_error(&nn, &features, &targets));

    MAT_PRINT(predictions);

    my_nn_train(&nn, &features, &targets, &hparams);

    printf("error: %lf\n", my_nn_calc_error(&nn, &features, &targets));

    my_nn_predict(&nn, &features, &predictions);

    MAT_PRINT(predictions);

    return 0;
}
