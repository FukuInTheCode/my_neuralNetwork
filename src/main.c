#include "../includes/my.h"

static double fun(double x)
{
    return x * x;
}

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

static void test_model(my_nn_t *nn, my_matrix_t *x, my_matrix_t *y, my_params_t *hp, double max, double min)
{

    MAT_DECLA(predicts);
    printf("--------------------------\n");
    printf("--------------------------\n");
    printf("--------%s-----\n", nn->name);
    printf("--------------------------\n");
    printf("--------------------------\n");
    srand(69);

    nn->size = 4;
    uint32_t dims[] = {x->m, 10, 10, y->m};

    my_nn_create(nn, dims);

    printf("starting error: %lf\n", my_nn_calc_error(nn, x, y));

    printf("took: %u\n", my_nn_train(nn, x, y, hp));

    printf("ending error: %lf\n", my_nn_calc_error(nn, x, y));

    my_nn_predict(nn, x, &predicts);
    my_matrix_transpose_2(&predicts);

    my_matrix_multiplybyscalar_2(&predicts, max);
    my_matrix_addscalar_2(&predicts, min);

    MAT_PRINT(predicts);
    MAT_FREE(predicts);
}

int main(int argc, char* argv[])
{
    srand(time(0));

    MAT_DECLA(features_tr);
    MAT_DECLA(features);
    MAT_DECLA(targets_tr);
    MAT_DECLA(targets);

    // my_matrix_create(4, 2, 1, &features_tr);
    // my_matrix_fill_from_array(&features_tr, xor_train_fea, 8);
    // my_matrix_create(4, 1, 1, &targets_tr);
    // my_matrix_fill_from_array(&targets_tr, xor_train_tar, 4);

    my_matrix_create(25, 1, 1, &features_tr);
    my_matrix_randint(-10, 10, 1, &features_tr);
    my_matrix_applyfunc(&features_tr, fun, &targets_tr);

    MAT_PRINT(features_tr);
    MAT_PRINT(targets_tr);

    double tmp_min = my_matrix_min(&features_tr);
    my_matrix_addscalar_2(&features_tr, -1. * tmp_min);
    double tmp_max = my_matrix_max(&features_tr);
    my_matrix_multiplybyscalar_2(&features_tr, 1. / tmp_max);

    double tmp_min_tar = my_matrix_min(&targets_tr);
    my_matrix_addscalar_2(&targets_tr, -1. * tmp_min_tar);
    double tmp_max_tar = my_matrix_max(&targets_tr);
    my_matrix_multiplybyscalar_2(&targets_tr, 1. / tmp_max_tar);

    my_matrix_transpose(&features_tr, &features);
    my_matrix_transpose(&targets_tr, &targets);

    printf("\n");
    printf("\n");
    printf("\n");
    printf("\n");
    printf("\n");
    printf("\n");

    // MAT_PRINT(features_tr);
    // MAT_PRINT(targets_tr);

    my_params_t hparams = {
        .alpha = 1e-1,
        .epoch = 50*1000,
        .threshold = 1e-3
    };

    my_nn_t neuro = {.name = "leaky"};

    // neuro.funcs.af = my_nn_leaky;
    // neuro.funcs.grad_af = my_nn_leaky_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    neuro.name = "sin";

    neuro.funcs.af = my_nn_sin;
    neuro.funcs.grad_af = my_nn_sin_grad;

    test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "relu";

    // neuro.funcs.af = my_nn_relu;
    // neuro.funcs.grad_af = my_nn_relu_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);
    // neuro.name = "softplus";

    // neuro.funcs.af = my_nn_softplus;
    // neuro.funcs.grad_af = my_nn_softplus_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);
    // neuro.name = "sigmoid";

    // neuro.funcs.af = my_nn_sigmoid;
    // neuro.funcs.grad_af = my_nn_sig_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);
    // neuro.name = "tanh";

    // neuro.funcs.af = my_nn_tanh;
    // neuro.funcs.grad_af = my_nn_tanh_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    my_nn_free(&neuro);

    return 0;
}
