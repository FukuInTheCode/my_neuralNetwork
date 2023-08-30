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

    MAT_DECLA(features);
    MAT_DECLA(targets);

    my_matrix_create(2, 100, 1, &features);
    my_matrix_create(1, 100, 1, &targets);

    MAT_PRINT_DIM(features);
    MAT_PRINT_DIM(targets);

    my_nn_t nn;

    nn.funcs.af = my_nn_sigmoid;
    nn.funcs.grad_af = my_nn_sig_grad;
    nn.size = 4;

    uint32_t dims[] = {2, 32, 32, 1};

    my_nn_create(&nn, dims, size);

    for (uint32_t i = 0; i < size - 1; ++i) {
        MAT_PRINT_DIM(nn.theta_arr[i]);
        MAT_PRINT_DIM(nn.bias_arr[i]);
    }

    my_nn_forward(&nn, &features, size);

    printf("\n");

    my_nn_backprogation(&nn, &features, &targets, size);

    return 0;
}
