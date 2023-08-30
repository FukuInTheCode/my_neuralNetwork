#include "../../includes/my.h"

double my_nn_relu(double x)
{
    return x < 0  ? 0 : x;
}
