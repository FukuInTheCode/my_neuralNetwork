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

    printf("--------------------------\n");
    printf("--------------------------\n");
    printf("--------%s-----\n", nn->name);
    printf("--------------------------\n");
    printf("--------------------------\n");
    srand(time(0));

    my_nn_create(nn);

    printf("starting error: %.10lf\n", my_nn_calc_error(nn, x, y));

    printf("%u\n", my_nn_train(nn, x, y, hp));

    printf("ending error: %.10lf\n", my_nn_calc_error(nn, x, y));

    MAT_DECLA(predicts);
    my_nn_predict(nn, x, &predicts);
    my_matrix_transpose_2(&predicts);

    if (max != 0)
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

    my_matrix_create(4, 2, 1, &features_tr);
    my_matrix_fill_from_array(&features_tr, xor_train_fea, 8);
    my_matrix_create(4, 1, 1, &targets_tr);
    my_matrix_fill_from_array(&targets_tr, xor_train_tar, 4);

    // my_matrix_create(25, 2, 1, &features_tr);
    // my_matrix_create(25, 1, 1, &targets_tr);
    // my_matrix_randint(10, -10, 1, &features_tr);
    // my_matrix_applyfunc(&features_tr, fun, &targets_tr);

    // for (uint32_t i = 0; i < targets_tr.m; ++i)
    //     my_matrix_set(&targets_tr, i, 0, features_tr.arr[i][0] * features_tr.arr[i][1]);

    MAT_PRINT(features_tr);
    MAT_PRINT(targets_tr);

    double tmp_min = my_matrix_min(&features_tr);
    my_matrix_addscalar_2(&features_tr, -1. * tmp_min);
    double tmp_max = my_matrix_max(&features_tr);
    if (tmp_max != 0)
        my_matrix_multiplybyscalar_2(&features_tr, 1. / tmp_max);

    double tmp_min_tar = my_matrix_min(&targets_tr);
    my_matrix_addscalar_2(&targets_tr, -1. * tmp_min_tar);
    double tmp_max_tar = my_matrix_max(&targets_tr);
    if (tmp_max_tar != 0)
        my_matrix_multiplybyscalar_2(&targets_tr, 1. / tmp_max_tar);

    my_matrix_transpose(&features_tr, &features);
    my_matrix_transpose(&targets_tr, &targets);

    my_params_t hparams = {
        .alpha = 1e-1,
        .epoch = 10*1000,
        .threshold = 1e-4,
        .show_tqdm = true
    };

    printf("aaa\n");

    NN_DECLA(neuro);
    NN_DECLA(cpy);

    neuro.size = 3;
    uint32_t dims[] = {features.m, 2, targets.m};

    neuro.dims = dims;

    my_nn_create(&neuro);

    my_nn_print(&neuro);

    double *arr = malloc(sizeof(double) * 6);
    my_nn_to_array(&neuro, &arr);

    for (uint32_t i = 0; i < 6; ++i)
        printf("%lf\n", arr[i]);

    // my_nn_copy(&neuro, &cpy);

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_sin;
    // neuro.funcs.grad_af = my_nn_sin_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_atan;
    // neuro.funcs.grad_af = my_nn_atan_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "soft sign";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_softsign;
    // neuro.funcs.grad_af = my_nn_softsign_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "sigmoid";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_sigmoid;
    // neuro.funcs.grad_af = my_nn_sigmoid_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "bin";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_binarystep;
    // neuro.funcs.grad_af = my_nn_binarystep_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "gelu";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_gelu;
    // neuro.funcs.grad_af = my_nn_gelu_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);


    // neuro.name = "idc";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_idc;
    // neuro.funcs.grad_af = my_nn_idc_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "leaky";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_leaky;
    // neuro.funcs.grad_af = my_nn_leaky_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "linear";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_linear;
    // neuro.funcs.grad_af = my_nn_linear_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "relu";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_relu;
    // neuro.funcs.grad_af = my_nn_relu_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "silu";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_silu;
    // neuro.funcs.grad_af = my_nn_silu_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "sinc";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_sinc;
    // neuro.funcs.grad_af = my_nn_sinc_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "soft plus";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_softplus;
    // neuro.funcs.grad_af = my_nn_softplus_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "tanh";

    // neuro.acti_type = base_type;
    // neuro.funcs.af = my_nn_tanh;
    // neuro.funcs.grad_af = my_nn_tanh_grad;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "elu";
    // neuro.acti_type = param_type;
    // neuro.funcs.af_p = my_nn_elu;
    // neuro.funcs.grad_af_p = my_nn_elu_grad;
    // double params[] = { 2. };
    // neuro.funcs.params = params;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "prelu";
    // neuro.acti_type = param_type;
    // neuro.funcs.af_p = my_nn_prelu;
    // neuro.funcs.grad_af_p = my_nn_prelu_grad;
    // double params3[] = { sqrt(2) };
    // neuro.funcs.params = params3;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "selu";
    // neuro.acti_type = param_type;
    // neuro.funcs.af_p = my_nn_selu;
    // neuro.funcs.grad_af_p = my_nn_selu_grad;
    // double params4[] = { 1.0507, 1.67326 };
    // neuro.funcs.params = params4;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // neuro.name = "soft exp";
    // neuro.acti_type = param_type;
    // neuro.funcs.af_p = my_nn_softexp;
    // neuro.funcs.grad_af_p = my_nn_softexp_grad;
    // double params5[] = { 0.138 };
    // neuro.funcs.params = params5;

    // test_model(&neuro, &features, &targets, &hparams, tmp_max_tar, tmp_min_tar);

    // my_nn_print(&neuro);
    // my_nn_print(&cpy);

    my_nn_free(&neuro);

    return 0;
}
