#include "../../includes/my.h"

double my_ln(double x, int n)
{
    if (x <= 0)
        return NAN;

    double result = 0.0;
    double term = (x - 1.0) / (x + 1.0);
    double current_term = term;

    for (int i = 1; i < n; i += 2) {
        result += current_term / i;
        current_term *= term * term;
    }

    return 2 * result;
}
