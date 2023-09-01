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

    printf("--------------------------\n");
    printf("--------------------------\n");
    printf("--------tanh-----\n");
    printf("--------------------------\n");
    printf("--------------------------\n");

    my_nn_t nn1;

    nn1.funcs.af = my_nn_tanh;
    nn1.funcs.grad_af = my_nn_tanh_grad;
    nn1.size = 2;

    uint32_t dims[] = {features.m, targets.m};

    my_nn_create(&nn1, dims);

    my_params_t hparams = {
        .alpha = 1e-2,
        .epoch = 10*1000,
        .threshold = 1e-10
    };

    printf("error: %lf\n", my_nn_calc_error(&nn1, &features, &targets));
    my_nn_predict(&nn1, &features, &predicts);
    MAT_PRINT(predicts);

    printf("took: %u\n", my_nn_train(&nn1, &features, &targets, &hparams));

    printf("error: %lf\n", my_nn_calc_error(&nn1, &features, &targets));

    my_nn_predict(&nn1, &features, &predicts);
    MAT_PRINT(predicts);

    printf("--------------------------\n");
    printf("--------------------------\n");
    printf("--------RELU-----\n");
    printf("--------------------------\n");
    printf("--------------------------\n");

    nn1.funcs.af = my_nn_relu;
    nn1.funcs.grad_af = my_nn_relu_grad;

    my_nn_create(&nn1, dims);

    printf("error: %lf\n", my_nn_calc_error(&nn1, &features, &targets));
    my_nn_predict(&nn1, &features, &predicts);
    MAT_PRINT(predicts);

    printf("took: %u\n", my_nn_train(&nn1, &features, &targets, &hparams));

    printf("error: %lf\n", my_nn_calc_error(&nn1, &features, &targets));

    my_nn_predict(&nn1, &features, &predicts);
    MAT_PRINT(predicts);

    my_nn_free(&nn1);

    return 0;
}
