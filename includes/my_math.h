#pragma once
#define MYMATHH

#include <math.h>

#define ll long long

static inline __attribute__((always_inline)) void swapint(int *a, int *b)
{
    const int temp = *a;
    *a = *b;
    *b = temp;
}

static inline __attribute__((always_inline)) void swap(double *a, double *b)
{
    const int temp = *a;
    *a = *b;
    *b = temp;
}


double my_dot_product(double* a, double* b, unsigned int size);
double my_power(double x, unsigned int n);
int my_randint(int minN, int maxN);
double my_randfloat(double minN, double maxN);
unsigned ll my_factorial(const unsigned int n);
double my_exp(double x, int n);
double my_ln(double x, int n);
double my_abs(double x);
int my_gcd(int a, int b);
