#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FatalError(str) fprintf(stderr, "%s\n", str), exit(1);
#define PI 3.141592654
#define DIMENSION 2
#define SIZE 100
#define EPSILON 0.0000001

//生成正态随机数
double GaussRand(double miu, double std)
{
    static double U, V;
    static int phase = 0;
    double Z;

    if (phase == 0)
    {
        U = rand() / (RAND_MAX + 1.0);
        V = rand() / (RAND_MAX + 1.0);
        Z = sqrt(-2.0 * log(U)) * sin(2.0 * PI * V);
    }
    else
    {
        Z = sqrt(-2.0 * log(U)) * cos(2.0 * PI * V);
    }

    phase = 1 - phase;
    return miu + std * Z;
}

//欧氏距离
double Euclidean(double* a, double* b)
{
    double total = 0;

    for (int i = 0; i < DIMENSION; i++)
        total += (a[i] - b[i]) * (a[i] - b[i]);

    return sqrt(total);
}

double EuclideanSquare(double* a, double* b)
{
    double total = 0;

    for (int i = 0; i < DIMENSION; i++)
        total += (a[i] - b[i]) * (a[i] - b[i]);

    return (total);
}

//高斯核
double GaussianKernel(double* a, double* b, double bandwidth)
{
    double total = 0;

    for (int i = 0; i < DIMENSION; i++)
        total += ((a[i] - b[i]) * (a[i] - b[i]));
    return exp(-1.0 / 2.0 * (total) / (bandwidth * bandwidth));
}

//打印数据
void PrintData(double* Point)
{
    int i;
    printf("坐标：(");
    for (i = 0; i < DIMENSION - 1; i++)
    {
        printf("%5.2f, ", Point[i]);
    }
    printf("%5.2f)\n", Point[i]);
}

void ShiftPoint(double* Point, double A[][DIMENSION], double Bandwidth)
{
    double Weights[SIZE];
    double Denominator = 0;
    int i, j;
    
    for (i = 0; i < SIZE; i++)
    {
        Weights[i] = GaussianKernel(Point, A[i], Bandwidth);
        Denominator += Weights[i];
    }
    for (j = 0; j < DIMENSION; j++)
        Point[j] = 0;
    for (i = 0; i < SIZE; i++)
        for (j = 0; j < DIMENSION; j++)
            Point[j] += Weights[i] * A[i][j] / Denominator;
}

void Cluster(double** Shift_Points, int* Clusters)
{
    int C = 0, i;

    for (i = 0; i < SIZE; i++)
    {
        Clusters[i] = 0;
    }
    for (i = 0; i < SIZE; i++)
    {
        if (Clusters[i] != 0)
            break;
        C += 1;
        Clusters[i] = C;
        for (int c = i; c < SIZE; c++)
            if (Euclidean(Shift_Points[i], Shift_Points[c]) <= EPSILON)
                Clusters[c] = C;
    }
}

int* MeanShift(double A[][DIMENSION], double Bandwidth)
{
    double** Shift_Points;
    double Tmp[DIMENSION];
    double Max_Dist = 1, Dist;
    int iter = 0, stop_shifting[SIZE] = { 0 }, i, j;
    int* Clusters;
    
    Shift_Points = (double**)malloc(SIZE * sizeof(double*));
    if (Shift_Points == NULL)
        FatalError("Out of space!!!");
    for (i = 0; i < SIZE; i++)
    {
        Shift_Points[i] = (double*)malloc(DIMENSION * sizeof(double));
        if (Shift_Points[i] == NULL)
            FatalError("Out of space!!!");
        for (j = 0; j < DIMENSION; j++)
            Shift_Points[i][j] = A[i][j];
    }

    while (Max_Dist > EPSILON)
    {
        Max_Dist = 0;
        iter += 1;
        for (i = 0; i < SIZE; i++)
        {
            if (stop_shifting[i])
                continue;
            for (j = 0; j < DIMENSION; j++)
            {
                Tmp[j] = Shift_Points[i][j];
            }
            ShiftPoint(Shift_Points[i], A, Bandwidth);
            Dist = Euclidean(Shift_Points[i], Tmp);
            if (Dist > Max_Dist)
                Max_Dist = Dist;
            if (Dist < EPSILON)
                stop_shifting[i] = 1;
        }
    }
    Clusters = malloc(SIZE * sizeof(int));
    if (Clusters == NULL)
        FatalError("Out of space!!!");
    Cluster(Shift_Points, Clusters);

    for (i = 0; i < SIZE; i++)
    {
        free(Shift_Points[i]);
    }
    free(Shift_Points);
    return Clusters;
}

int* BlurringMeanShift(double A[][DIMENSION], double Bandwidth)
{
    double** Shift_Points;
    double Tmp[DIMENSION];
    double Max_Dist = 1, Dist;
    int iter = 0, stop_shifting[SIZE] = { 0 }, i, j;
    int* Clusters;

    Shift_Points = (double**)malloc(SIZE * sizeof(double*));
    if (Shift_Points == NULL)
        FatalError("Out of space!!!");
    for (i = 0; i < SIZE; i++)
    {
        Shift_Points[i] = (double*)malloc(DIMENSION * sizeof(double));
        if (Shift_Points[i] == NULL)
            FatalError("Out of space!!!");
        for (j = 0; j < DIMENSION; j++)
            Shift_Points[i][j] = A[i][j];
    }

    while (Max_Dist > EPSILON)
    {
        Max_Dist = 0;
        iter += 1;
        for (i = 0; i < SIZE; i++)
        {
            if (stop_shifting[i])
                continue;
            for (j = 0; j < DIMENSION; j++)
            {
                Tmp[j] = Shift_Points[i][j];
            }
            ShiftPoint(Shift_Points[i], A, Bandwidth);
            Dist = Euclidean(Shift_Points[i], Tmp);
            if (Dist > Max_Dist)
                Max_Dist = Dist;
            if (Dist < EPSILON)
                stop_shifting[i] = 1;
        }
        for (i = 0; i < SIZE; i++)
        {
            if (stop_shifting[i])
                continue;
            for (j = 0; j < DIMENSION; j++)
            {
                A[i][j] = Shift_Points[i][j];
            }
        }
    }
    Clusters = malloc(SIZE * sizeof(int));
    if (Clusters == NULL)
        FatalError("Out of space!!!");
    Cluster(Shift_Points, Clusters);

    for (i = 0; i < SIZE; i++)
    {
        free(Shift_Points[i]);
    }
    free(Shift_Points);
    return Clusters;
}

int main()
{
    int i, j;
    double A[SIZE][DIMENSION];
    int* Clusters;
    int* BlurringClusters;

    //生成正态分布随机数
    for (i = 0; i < 100; i++)
        if (i < 50)
            for (j = 0; j < DIMENSION; j++)
                A[i][j] = GaussRand(0, 0.5);
        else
            for (j = 0; j < DIMENSION; j++)
                A[i][j] = GaussRand(2, 0.5);
    
    printf("Meanshift的聚类结果：\n");
    Clusters = MeanShift(A, 0.6);
    //打印生成的数据
    for (i = 0; i < SIZE; i++)
    {
        printf("点的坐标：");
        for (j = 0; j < DIMENSION; j++)
        {
            printf("%5.2f ", A[i][j]);
        }
        printf("  类=%d\n", Clusters[i]);
    }
    free(Clusters);
    BlurringClusters = BlurringMeanShift(A, 0.4);
    printf("Blurring meanshift的聚类结果：\n");
    for (i = 0; i < SIZE; i++)
    {
        printf("点的坐标：");
        for (j = 0; j < DIMENSION; j++)
        {
            printf("%5.2f ", A[i][j]);
        }
        printf("  类=%d\n", BlurringClusters[i]);
    }
    free(BlurringClusters);
    return 0;
}