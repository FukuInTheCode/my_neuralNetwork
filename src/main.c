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
    srand(69);
    my_matrix_t features = {.m = 0, .n = 0, .name = "features"};
    my_matrix_t targets = {.m = 0, .n = 0, .name = "targets"};
    my_matrix_t predictions = {.m = 0, .n = 0, .name = "predictions"};
    my_matrix_create(25, 1, 1, &features);
    my_matrix_randfloat(1, 100, 1, &features);
    // my_matrix_fill_from_array(&features, unknw_train_fea, 8);
    // my_matrix_create(4, 1, 1, &targets);
    // my_matrix_fill_from_array(&targets, unknw_train_tar, 4);
    my_matrix_applyfunc(&features, log, &targets);
    my_matrix_print(2, &features, &targets);

    uint32_t layers[] = {features.n, 10, 10, targets.n};
    my_nn_t nn = {.layers = layers, .layers_size = sizeof(layers) / sizeof(uint32_t), .name = "Neuro", .apply_all = my_false};
    my_nn_create(&nn);
    my_nn_print(&nn);

    my_nn_predict(&nn, &features, &predictions);
    my_matrix_print(1, &predictions);
    printf("Starting Error: %lf\n", my_nn_calcerror_mse(&nn, &features, &targets));

    my_nn_backpropagation(&nn, &features, &targets);

    my_params_t p = {
        .iterations = 1000*100,
        .alpha = 1e-1,
        .threshold = 1e-5
    };

    my_nn_train(&nn, &features, &targets, &p);
    my_nn_print(&nn);
    printf("Finishing Error: %lf\n", my_nn_calcerror_mse(&nn, &features, &targets));
    my_nn_predict(&nn, &features, &predictions);
    my_matrix_print(1, &predictions);

    my_matrix_free(2, &features, &targets);
    my_nn_free(&nn);
    return 0;
}
