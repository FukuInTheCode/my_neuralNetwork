#include "../includes/my.h"

int main(int argc, char* argv[]) {
    my_Matrix A = {.m=0, .n=0};
    my_Matrix_Create(10, 10, 1, &A);
    my_Matrix_RandFloat(0, 10, 1, &A);
    my_Matrix_Print(1, &A);
    my_Matrix_Free(1, &A);
    return 0;
}