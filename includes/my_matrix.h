#pragma once
#define MYMATRIXH

#include "./my_math.h"

typedef struct my_Matrix {
    unsigned int m;
    unsigned int n;
    double **arr;
} my_Matrix;

typedef double (*UnaryFunction)(double);

void my_Matrix_Create(unsigned int m, unsigned int n, const unsigned int count, ...);
void my_Matrix_Identity(const unsigned int count, ...);
void my_Matrix_Set(my_Matrix* A, unsigned int x, unsigned int y, double n);
void my_Matrix_Print(const unsigned int count, ...);
void my_Matrix_Free(const unsigned int count, ...);
void my_Matrix_MultiplyByScalar(my_Matrix* A, double scalar, my_Matrix* result);
void my_Matrix_Add(my_Matrix* result, const unsigned int count, ...);
void my_Matrix_T(my_Matrix* A, my_Matrix* T);
void my_Matrix_GetColumn(my_Matrix* A, unsigned int n, double result[]);
double* my_Matrix_GetRow(my_Matrix* A, unsigned int i);
void my_Matrix_Product(my_Matrix* result, const unsigned int count, ...);
void my_Matrix_Copy(my_Matrix* A, my_Matrix* copy);
void my_Matrix_PowerInt(my_Matrix* A, const unsigned int n, my_Matrix* result);
void my_Matrix_RandInt(int minN, int maxN, const unsigned int count, ...);
void my_Matrix_RandFloat(double minN, double maxN, const unsigned int count, ...);
double my_Matrix_Det(my_Matrix *A);
void my_Matrix_GetSubMatrix(my_Matrix *A, const unsigned int m, const unsigned int n, my_Matrix *result);
void my_Matrix_Adjugate(my_Matrix *A, my_Matrix *result);
void my_Matrix_Inverse(my_Matrix *A, my_Matrix *result);
void my_Matrix_AddScalar(my_Matrix* A, double scalar, my_Matrix* result);
void my_Matrix_ApplyFunc(my_Matrix *A, UnaryFunction func, my_Matrix *result);
double my_Matrix_Sum(my_Matrix *A);
void my_Matrix_One(my_Matrix *A, my_Matrix *result);
void my_Matrix_AddCol(my_Matrix *A, const unsigned int n, my_Matrix *result);
void my_Matrix_AddRow(my_Matrix *A, const unsigned int m, my_Matrix *result);
void my_Matrix_ConcatCol(my_Matrix *result, my_Matrix *A, my_Matrix *B);
void my_Matrix_ConcatRow(my_Matrix *result, my_Matrix *A, my_Matrix *B);
int my_Matrix_Equals(my_Matrix *A, my_Matrix *B);
void my_Matrix_SwapRow(my_Matrix *A, const unsigned int a, const unsigned int b, my_Matrix *result);
void my_Matrix_SwapCol(my_Matrix *A, const unsigned int a, const unsigned int b, my_Matrix *result);
void my_Matrix_SumCol(my_Matrix *A, my_Matrix *result);
void my_Matrix_SumRow(my_Matrix *A, my_Matrix *result);
void my_Matrix_SetRow(my_Matrix *A, const unsigned int m, const double x);
void my_Matrix_SetCol(my_Matrix *A, const unsigned int n, const double x);
void my_Matrix_Broadcasting(my_Matrix *A, const unsigned int m, const unsigned int n, my_Matrix *result); 
double my_Matrix_Max(my_Matrix *A);
double my_Matrix_Min(my_Matrix *A);
void my_Matrix_ElementsMultiplication(my_Matrix* result, const unsigned int count, ...);
double my_Matrix_MaxCol(my_Matrix *A, const unsigned int n);
double my_Matrix_MaxRow(my_Matrix *A, const unsigned int m);
double my_Matrix_MinCol(my_Matrix *A, const unsigned int n);
double my_Matrix_MinRow(my_Matrix *A, const unsigned int m);