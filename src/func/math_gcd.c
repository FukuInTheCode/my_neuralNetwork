#include "../../includes/my.h"


int my_gcd(int a, int b)
{
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }

    return a;
}
