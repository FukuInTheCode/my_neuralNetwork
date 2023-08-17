#include "../../includes/my.h"

static void create_arrelement(my_matrix_t *A, unsigned int m, unsigned int n)
{
    unsigned int i;

    A->m = m;
    A->n = n;
    A->arr = malloc(m * sizeof(double *));
    if (A->arr == NULL) {
        fprintf(stderr, "2Memory allocation failed.\n");
        exit(1);
    }
    for (i = 0; i < m; i++) {
        A->arr[i] = calloc(n, sizeof(double));
        if (A->arr[i] == NULL) {
            fprintf(stderr, "3Memory allocation failed.\n");
            exit(1);
        }
    }
}

void my_nn_create(my_nn_t *N)
{
    N->theta_arr = malloc((N->layers_size - 1) * sizeof(my_matrix_t));
    N->bias_arr = malloc((N->layers_size - 1) * sizeof(my_matrix_t));
    if (N->theta_arr == NULL || N->bias_arr == NULL) {
        fprintf(stderr, "1Memory allocation failed.\n");
        exit(1);
    }
    uint8_t m = N->layers[0];
    for (uint8_t i = 1; i < N->layers_size; i++) {
        uint8_t n = N->layers[i];
        create_arrelement(&(N->theta_arr[i - 1]), m, n);
        create_arrelement(&(N->bias_arr[i - 1]), 1, n);
        my_matrix_randfloat(-1, 1, 2, &(N->theta_arr[i - 1]),\
                                        &(N->bias_arr[i - 1]));
        m = n;
    }
    my_nn_create_activation(N, N->layers[0]);
    my_nn_create_gradients(N);
}

void my_nn_create_activation(my_nn_t *N, uint8_t inputs_size)
{
    N->activations = malloc(N->layers_size * sizeof(my_matrix_t));
    if (N->activations == NULL) {
        fprintf(stderr, "1Memory allocation failed.\n");
        exit(1);
    }
    uint8_t n = N->theta_arr[0].m;
    for (uint8_t i = 0; i < N->layers_size; i++) {
        create_arrelement(&(N->activations[i]), inputs_size, n);
        n = N->theta_arr[i].n;
    }
}

void my_nn_create_gradients(my_nn_t *N)
{
    N->gradientsTheta = malloc((N->layers_size - 1) * sizeof(my_matrix_t));
    N->gradientsBias = malloc((N->layers_size - 1) * sizeof(my_matrix_t));
    if (N->gradientsBias == NULL || N->gradientsTheta == NULL) {
        fprintf(stderr, "1Memory allocation failed.\n");
        exit(1);
    }
    for (uint8_t i = 0; i < N->layers_size - 1; i++) {
        create_arrelement(&(N->gradientsBias[i]), N->bias_arr[i].m, N->bias_arr[i].n);
        create_arrelement(&(N->gradientsTheta[i]), N->theta_arr[i].m, N->theta_arr[i].n);
    }
}
