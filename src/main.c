#include "../includes/my.h"

static double my_func(double x){
    return x * x;
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    my_matrix_t features = {.m = 0, .n = 0, .name = "features"};
    my_matrix_t targets = {.m = 0, .n = 0, .name = "targets"};
    my_matrix_create(2, 2, 1, &features);
    my_matrix_randint(-10, 10, 1, &features);
    my_matrix_applyfunc(&features, my_func, &targets);
    my_matrix_print(2, &features, &targets);
    my_matrix_free(2, &features, &targets);
    return 0;
}
