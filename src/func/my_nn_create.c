#include "../../includes/my.h"

static void create_arrElement(my_matrix_t *A, unsigned int m, unsigned int n)
{
    unsigned int i;

    A->m = m;
    A->n = n;
    A->arr = malloc(m * sizeof(double *));
    if (A->arr == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    for (i = 0; i < m; i++) {
        A->arr[i] = calloc(n, sizeof(double));
        if (A->arr[i] == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(1);
        }
    }
}

void my_nn_create(my_nn_t *N, uint8_t layers[], \
    const uint8_t layers_size)
{
    unsigned int i;
    uint8_t m = layers[0];
    N->theta_arr = malloc((layers_size - 1) * sizeof(my_matrix_t));
    N->bias_arr = malloc((layers_size - 1) * sizeof(my_matrix_t));
    if (N->theta_arr == NULL || N->bias_arr == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    for (i = 1; i < layers_size; i++) {
        uint8_t n = layers[i];
        create_arrElement(&(N->theta_arr[i - 1]), m, n);
        create_arrElement(&(N->bias_arr[i - 1]), m, 1);
        my_matrix_randfloat(-1, 1, 2, &(N->theta_arr[i - 1]), &(N->bias_arr[i - 1]));
        my_matrix_print(2, &(N->theta_arr[i - 1]), &(N->bias_arr[i - 1]));
        m = n;
    }
}