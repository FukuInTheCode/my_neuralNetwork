#pragma once
#define MYH

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include "my_neuralnetwork.h"

static inline __attribute__((always_inline)) void check_alloc(void *A)
{
    if (A == NULL) {
        fprintf(stderr, "Memory allocation failed!");
        exit(1);
    }
}
