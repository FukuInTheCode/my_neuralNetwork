#include "../../includes/my.h"

unsigned ll my_factorial(const unsigned int n)
{
    unsigned int i;

    unsigned ll res = 1;
    for (i = 1; i <= n; i++)
        res *= i;

    return res;
}
