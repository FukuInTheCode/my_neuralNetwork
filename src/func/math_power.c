#include "../../includes/my.h"

double my_power(double x, unsigned int n)
{
    unsigned int i;

    int res = 1;
    for (i = 0; i < n; i++)
        res *= x;

    return res;
}
