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

static void test_model(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y, my_params_t *hp)
{

    MAT_DECLA(predicts);
    printf("--------------------------\n");
    printf("--------------------------\n");
    printf("--------%s-----\n", nn->name);
    printf("--------------------------\n");
    printf("--------------------------\n");
    srand(69);

    nn->size = 3;
    uint32_t dims[] = {x->m, 2, y->m};

    my_nn_create(nn, dims);

    printf("starting error: %lf\n", my_nn_calc_error(nn, x, y));

    printf("took: %u\n", my_nn_train(nn, x, y, hp));

    printf("ending error: %lf\n", my_nn_calc_error(nn, x, y));

    my_nn_predict(nn, x, &predicts);
    MAT_PRINT(predicts);
    MAT_FREE(predicts);
}

int main(int argc, char* argv[])
{
    srand(69);

    MAT_DECLA(features_tr);
    MAT_DECLA(features);
    MAT_DECLA(targets_tr);
    MAT_DECLA(targets);
    MAT_DECLA(predicts);

    my_matrix_create(4, 2, 1, &features_tr);
    my_matrix_fill_from_array(&features_tr, xor_train_fea, 8);
    my_matrix_create(4, 1, 1, &targets_tr);
    my_matrix_fill_from_array(&targets_tr, xor_train_tar, 4);

    my_matrix_transpose(&features_tr, &features);
    my_matrix_transpose(&targets_tr, &targets);

    MAT_PRINT(features_tr);
    MAT_PRINT(targets_tr);

    my_params_t hparams = {
        .alpha = 1e-1,
        .epoch = 100*1000,
        .threshold = 1e-3
    };

    my_nn_t neuro = {.name = "neuro"};

    neuro.funcs.af = my_nn_sin;
    neuro.funcs.grad_af = my_nn_sin_grad;

    test_model(&neuro, &features, &targets, &hparams);

    my_nn_free(&neuro);

    return 0;
}
