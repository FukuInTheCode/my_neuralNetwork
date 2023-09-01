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
    return x * x;
}

int main(int argc, char* argv[])
{
    srand(time(0));

    MAT_DECLA(features_tr);
    MAT_DECLA(features);
    MAT_DECLA(targets_tr);
    MAT_DECLA(targets);
    MAT_DECLA(predicts);

    my_matrix_create(4, 2, 1, &features_tr);
    my_matrix_fill_from_array(&features_tr, unknw_train_fea, 8);
    my_matrix_create(4, 1, 1, &targets_tr);
    my_matrix_fill_from_array(&targets_tr, unknw_train_tar, 4);

    my_matrix_transpose(&features_tr, &features);
    my_matrix_transpose(&targets_tr, &targets);

    MAT_PRINT(features_tr);
    MAT_PRINT(targets_tr);

    my_nn_t nn;

    nn.funcs.af = my;
    nn.funcs.grad_af = my_nn_sig_grad;
    nn.size = 2;

    uint32_t dims[] = {features.m, targets.m};

    my_nn_create(&nn, dims);

    my_params_t hparams = {
        .alpha = 1e-2,
        .epoch = 10*1000,
        .threshold = 1e-10
    };

    printf("error: %lf\n", my_nn_calc_error(&nn, &features, &targets));
    my_nn_predict(&nn, &features, &predicts);
    MAT_PRINT(predicts);

    my_nn_train(&nn, &features, &targets, &hparams);

    printf("error: %lf\n", my_nn_calc_error(&nn, &features, &targets));

    my_nn_predict(&nn, &features, &predicts);
    MAT_PRINT(predicts);

    my_nn_free(&nn);

    return 0;
}
