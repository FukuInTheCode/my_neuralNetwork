#include "../includes/my.h"

int main(int argc, char* argv[])
{
    srand(time(NULL));
    uint8_t m = 50;
    my_matrix_t features = {.m = 0, .n = 0};
    my_matrix_t subfeatures1 = {.m = 0, .n = 0};
    my_matrix_t subfeatures2 = {.m = 0, .n = 0};
    my_matrix_t targets = {.m = 0, .n = 0};
    printf("train set: \n");
    my_matrix_create(m, 1, 2, &subfeatures1, &subfeatures2);
    my_matrix_randint(0, 10, 2, &subfeatures1, &subfeatures2);
    my_matrix_concatcol(&features, &subfeatures1, &subfeatures2);
    my_matrix_product_elementwise(&targets, 2, &subfeatures1, &subfeatures2);
    my_matrix_print(2, &features, &targets);
    my_matrix_t features_test = {.m = 0, .n = 0};
    my_matrix_t subfeatures1_test = {.m = 0, .n = 0};
    my_matrix_t subfeatures2_test = {.m = 0, .n = 0};
    my_matrix_t targets_test = {.m = 0, .n = 0};
    printf("test set: \n");
    my_matrix_create(m, 1, 2, &subfeatures1_test, &subfeatures2_test);
    my_matrix_randint(0, 10, 2, &subfeatures1_test, &subfeatures2_test);
    my_matrix_concatcol(&features_test, &subfeatures1_test, &subfeatures2_test);
    my_matrix_product_elementwise(&targets_test, 2, &subfeatures1_test, &subfeatures2_test);
    my_matrix_print(2, &features_test, &targets_test);

    printf("-------------------------\n");
    // uint32_t layers[] = {2, 1};
    // my_nn_t N = {.layers_size = 2, .layers = layers};
    // uint32_t layers[] = {2, 3, 1};
    // my_nn_t N = {.layers_size = 3, .layers = layers};
    uint32_t layers[] = {2, 32, 32, 1};
    my_nn_t N = {.layers_size = 4, .layers = layers};
    // uint32_t layers[] = {2, 4, 4, 1};
    // my_nn_t N = {.layers_size = 4, .layers = layers};
    // uint32_t layers[] = {2, 4, 8, 16, 32, 32, 16, 8, 4, 1};
    // my_nn_t N = {.layers_size = 10, .layers = layers};
    my_nn_create(&N);

    printf("predictions train set\n");
    my_nn_forwardpropagation(&N, &features);
    my_matrix_print(1, &(N.activations[N.layers_size - 1]));
    printf("ERROR train set: %f\n", my_nn_calcerror_mse(&N, &features, &targets));
    printf("predictions test set\n");
    my_nn_forwardpropagation(&N, &features_test);
    my_matrix_print(1, &(N.activations[N.layers_size - 1]));
    printf("ERROR test set: %f\n", my_nn_calcerror_mse(&N, &features_test, &targets_test));

    my_params_t parameters = {.alpha = 1e-4, .iterations = 50000, .threshold = 1e-1};

    my_nn_train(&N, &features, &targets, &parameters);

    printf("predictions train set\n");
    my_matrix_print(1, &(N.activations[N.layers_size - 1]));
    printf("ERROR train set: %f\n", my_nn_calcerror_mse(&N, &features, &targets));
    printf("predictions test set\n");
    my_nn_forwardpropagation(&N, &features_test);
    my_matrix_print(1, &(N.activations[N.layers_size - 1]));
    printf("ERROR test set: %f\n", my_nn_calcerror_mse(&N, &features_test, &targets_test));

    my_nn_free(&N);
    my_matrix_free(7, &features, &targets, &subfeatures1, &subfeatures2, &subfeatures1_test, &subfeatures2_test, &targets_test);
    return 0;
}
