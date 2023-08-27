#include "../includes/my.h"

static double my_func(double x){
    return x * x;
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    my_matrix_t features = {.m = 0, .n = 0, .name = "features"};
    my_matrix_t targets = {.m = 0, .n = 0, .name = "targets"};
    my_matrix_t predictions = {.m = 0, .n = 0, .name = "predictions"};
    my_matrix_create(4, 2, 1, &features);
    my_matrix_randint(0, 1, 1, &features);
    my_matrix_applyfunc(&features, my_func, &targets);
    my_matrix_print(2, &features, &targets);

    uint32_t layers[] = {features.n, 2, 2, targets.n};
    my_nn_t nn = {.layers = layers, .layers_size = 4, .name = "Neuro"};
    my_nn_create(&nn);
    my_nn_print(&nn);

    my_nn_predict(&nn, &features, &predictions);
    my_matrix_print(1, &predictions);
    printf("Starting Error: %lf\n", my_nn_calcerror_mse(&nn, &features, &targets));

    my_params_t p = {
        10000,
        1e-2,
        1e-2
    };

    my_nn_train(&nn, &features, &targets, &p);
    printf("%u, %u\n", predictions.m, predictions.n);
    printf("%f, %f\n", predictions.arr[0][0], predictions.arr[1][0]);
    my_nn_print(&nn);
    printf("Finishing Error: %lf\n", my_nn_calcerror_mse(&nn, &features, &targets));
    my_nn_predict(&nn, &features, &predictions);
    my_matrix_print(1, &predictions);

    my_matrix_free(2, &features, &targets);
    my_nn_free(&nn);
    return 0;
}
