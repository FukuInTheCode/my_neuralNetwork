#include "../../includes/my.h"

uint32_t my_nn_get_n_params(my_nn_t *nn)
{
    uint32_t res = 0;
    for (uint32_t i = 0; i < nn->size - 1; ++i) {
        res += nn->bias_arr[i].m * nn->bias_arr[i].n;
        res += nn->theta_arr[i].m * nn->theta_arr[i].n;
    }
}
