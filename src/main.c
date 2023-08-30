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

    my_matrix_create(2, 100, 1, &features);

    MAT_PRINT_DIM(features);

    my_nn_t nn;

    nn.funcs.af = my_nn_sigmoid;

    uint32_t size = 4;

    uint32_t dims[] = {2, 32, 32, 1};

    my_nn_create(&nn, dims, size);

    my_nn_forward(&nn, &features, size);

    return 0;
}
