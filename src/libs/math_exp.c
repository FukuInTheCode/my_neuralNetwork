#include "../../includes/my.h"

double my_exp(double x, int n)
{
    double result = 1.0;
    double term = 1.0;

    for (int i = 1; i < n; ++i) {
        term *= x / i;
        result += term;
    }

    return result;
}
