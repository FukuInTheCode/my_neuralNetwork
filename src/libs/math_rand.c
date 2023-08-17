#include "../../includes/my.h"

int my_randint(int minN, int maxN)
{
    maxN++;
    if (minN > maxN)
        swapint(&minN, &maxN);

    return rand() % (maxN - minN) + minN;
}

double my_randfloat(double minN, double maxN)
{
    if ( minN > maxN)
        swap(&minN, &maxN);

    double range = (maxN - minN);
    double div = RAND_MAX / range;

    return minN + (rand() / div);
}
