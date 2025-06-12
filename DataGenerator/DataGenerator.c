#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_N 1000
#define STEP_N 10
#define MAX_THREADS 12
#define DET_MAX_SIZE 12


typedef enum {
    RANDOM_DENSE = 0,
    SPARSE_50 = 1,
    DIAGONAL_DOMINANT = 2
} MatrixType;


void generateMatrix(int n, int **matrix, MatrixType type) {
    switch(type) {
        case RANDOM_DENSE:
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    matrix[i][j] = rand() % 10;
                }
            }
            break;

        case SPARSE_50:
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    matrix[i][j] = (rand() % 2 == 0) ? 0 : (rand() % 10);
                }
            }
            break;

        case DIAGONAL_DOMINANT:
            for (int i = 0; i < n; i++) {
                int sum = 0;
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        matrix[i][j] = rand() % 5;
                        sum += abs(matrix[i][j]);
                    }
                }
                matrix[i][i] = sum + (rand() % 10) + 1;
            }
            break;
    }
}


void luDecomposition(int n, int **A, int **L, int **U, int *P, int num_threads) {

    omp_set_num_threads(num_threads);


    #pragma omp parallel for collapse(2) if(num_threads > 1)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            U[i][j] = A[i][j];
            L[i][j] = (i == j) ? 1 : 0;
        }
        P[i] = i;
    }

    for (int k = 0; k < n; k++) {

        int pivot = k;
        for (int i = k + 1; i < n; i++) {
            if (abs(U[i][k]) > abs(U[pivot][k])) {
                pivot = i;
            }
        }


        if (pivot != k) {

            for (int j = 0; j < n; j++) {
                int temp = U[k][j];
                U[k][j] = U[pivot][j];
                U[pivot][j] = temp;
            }


            for (int j = 0; j < k; j++) {
                int temp = L[k][j];
                L[k][j] = L[pivot][j];
                L[pivot][j] = temp;
            }


            int temp = P[k];
            P[k] = P[pivot];
            P[pivot] = temp;
        }


        #pragma omp parallel for if(num_threads > 1)
        for (int i = k + 1; i < n; i++) {
            L[i][k] = U[i][k] / U[k][k];
            for (int j = k; j < n; j++) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }
    }
}


void powerMethod(int n, int **A, double *eigenvalue, double **eigenvector, int max_iter, double tol, int num_threads) {

    double *v = (double*)malloc(n * sizeof(double));
    double *v_new = (double*)malloc(n * sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        v[i] = (double)rand() / RAND_MAX;
    }


    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        norm += v[i] * v[i];
    }
    norm = sqrt(norm);

    for (int i = 0; i < n; i++) {
        v[i] /= norm;
    }

    double lambda_old = 0.0;
    double lambda = 0.0;

    for (int iter = 0; iter < max_iter; iter++) {

        omp_set_num_threads(num_threads);
        #pragma omp parallel for if(num_threads > 1)
        for (int i = 0; i < n; i++) {
            v_new[i] = 0.0;
            for (int j = 0; j < n; j++) {
                v_new[i] += A[i][j] * v[j];
            }
        }


        lambda_old = lambda;
        lambda = 0.0;
        for (int i = 0; i < n; i++) {
            if (fabs(v_new[i]) > fabs(lambda)) {
                lambda = v_new[i];
            }
        }


        if (fabs(lambda - lambda_old) < tol) {
            break;
        }


        norm = 0.0;
        for (int i = 0; i < n; i++) {
            v_new[i] /= lambda;
            norm += v_new[i] * v_new[i];
        }
        norm = sqrt(norm);

        for (int i = 0; i < n; i++) {
            v[i] = v_new[i] / norm;
        }
    }


    *eigenvalue = lambda;
    for (int i = 0; i < n; i++) {
        (*eigenvector)[i] = v[i];
    }

    free(v);
    free(v_new);
}


void matrixMultiply(int N, int **A, int **B, int **C, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2) schedule(static, 10)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


void matrixAdd(int N, int **A, int **B, int **C, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2) schedule(static, 10)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}


void matrixTranspose(int N, int **A, int **T, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2) schedule(static, 10)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            T[j][i] = A[i][j];
        }
    }
}

void matrixScale(int N, int **A, int scalar, int **Result, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2) schedule(static, 10) if(num_threads > 1)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Result[i][j] = A[i][j] * scalar;
        }
    }
}

void matrixExponential(int N, int **A, double **Result, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2) schedule(static, 10) if(num_threads > 1)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Result[i][j] = exp((double)A[i][j]);
        }
    }
}

void matrixLogarithm(int N, int **A, double **Result, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2) schedule(static, 10) if(num_threads > 1)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            Result[i][j] = log((double)A[i][j] + 1.0);
        }
    }
}

void matrixSqrt(int N, int **A, double **Result, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2) schedule(static, 10) if(num_threads > 1)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Result[i][j] = sqrt((double)A[i][j]);
        }
    }
}


void getSubmatrix(int n, int **matrix, int **temp, int skip_row, int skip_col) {
    int r = 0, c = 0;
    for (int i = 0; i < n; i++) {
        if (i == skip_row) continue;
        c = 0;
        for (int j = 0; j < n; j++) {
            if (j == skip_col) continue;
            temp[r][c] = matrix[i][j];
            c++;
        }
        r++;
    }
}



int determinantRecursive(int n, int **matrix) {

    if (n == 1) return matrix[0][0];


    if (n == 2)
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

    int det = 0;
    int sign = 1;
    int **temp = (int**)malloc((n-1) * sizeof(int*));
    for (int i = 0; i < n-1; i++)
        temp[i] = (int*)malloc((n-1) * sizeof(int));


    for (int i = 0; i < n; i++) {
        getSubmatrix(n, matrix, temp, 0, i);
        det += sign * matrix[0][i] * determinantRecursive(n-1, temp);
        sign = -sign;
    }


    for (int i = 0; i < n-1; i++)
        free(temp[i]);
    free(temp);

    return det;
}



double determinantGaussian(int n, int **matrix, int num_threads) {

    double **A = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        A[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)matrix[i][j];
        }
    }

    double det = 1.0;


    for (int i = 0; i < n; i++) {
        int max_row = i;
        for (int j = i + 1; j < n; j++) {
            if (fabs(A[j][i]) > fabs(A[max_row][i])) {
                max_row = j;
            }
        }

        if (max_row != i) {

            double *temp = A[i];
            A[i] = A[max_row];
            A[max_row] = temp;
            det = -det;
        }


        if (fabs(A[i][i]) < 1e-10) {
            for (int k = 0; k < n; k++) {
                free(A[k]);
            }
            free(A);
            return 0;
        }

        det *= A[i][i];


        #pragma omp parallel for num_threads(num_threads) if(num_threads > 1)
        for (int j = i + 1; j < n; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = i + 1; k < n; k++) {
                A[j][k] -= factor * A[i][k];
            }
        }
    }


    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);

    return det;
}


double determinantSmart(int n, int **matrix, int num_threads) {
    if (n <= DET_MAX_SIZE) {
        return determinantRecursive(n, matrix);
    } else {
        return determinantGaussian(n, matrix, num_threads);
    }
}


int** allocateMatrix(int N) {
    int **matrix = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        matrix[i] = (int*)malloc(N * sizeof(int));
    }
    return matrix;
}


void freeMatrix(int **matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}


int findOptimalThreadCount(int N, int **A, int **B, int **C, int **T, char* opType) {
    double min_time = DBL_MAX;
    int optimal_threads = 1;


    int **L = allocateMatrix(N);
    int **U = allocateMatrix(N);
    int *P = (int*)malloc(N * sizeof(int));
    double *eigenvalue = (double*)malloc(sizeof(double));
    double **eigenvector = (double**)malloc(sizeof(double*));
    *eigenvector = (double*)malloc(N * sizeof(double));


    double **D = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        D[i] = (double*)malloc(N * sizeof(double));
    }

    for (int threads = 1; threads <= MAX_THREADS; threads++) {
        double start, end, time_taken;


        if (strcmp(opType, "Multiplication") == 0) {
            start = omp_get_wtime();
            matrixMultiply(N, A, B, C, threads);
            end = omp_get_wtime();
        }
        else if (strcmp(opType, "Addition") == 0) {
            start = omp_get_wtime();
            matrixAdd(N, A, B, C, threads);
            end = omp_get_wtime();
        }
        else if (strcmp(opType, "Transposition") == 0) {
            start = omp_get_wtime();
            matrixTranspose(N, A, T, threads);
            end = omp_get_wtime();
        }
        else if (strcmp(opType, "Determinant") == 0) {
            start = omp_get_wtime();
            double det = determinantSmart(N, A, threads);
            end = omp_get_wtime();
        }
        else if (strcmp(opType, "LUDecomposition") == 0) {
            start = omp_get_wtime();
            luDecomposition(N, A, L, U, P, threads);
            end = omp_get_wtime();
        }
        else if (strcmp(opType, "Eigenvalue") == 0) {
            start = omp_get_wtime();
            powerMethod(N, A, eigenvalue, eigenvector, 100, 1e-6, threads);
            end = omp_get_wtime();
        }
        else if (strcmp(opType, "Scaling") == 0) {
            start = omp_get_wtime();
            matrixScale(N, A, 2, C, threads);
            end = omp_get_wtime();
        }
        else if (strcmp(opType, "Exponential") == 0) {
            start = omp_get_wtime();
            matrixExponential(N, A, D, threads);
            end = omp_get_wtime();
        }
        else if (strcmp(opType, "Logarithm") == 0) {
            start = omp_get_wtime();
            matrixLogarithm(N, A, D, threads);
            end = omp_get_wtime();
        }
        else if (strcmp(opType, "SquareRoot") == 0) {
            start = omp_get_wtime();
            matrixSqrt(N, A, D, threads);
            end = omp_get_wtime();
        }

        time_taken = end - start;


        if (time_taken < min_time) {
            min_time = time_taken;
            optimal_threads = threads;
        }
    }


    freeMatrix(L, N);
    freeMatrix(U, N);
    free(P);
    free(eigenvalue);
    free(*eigenvector);
    free(eigenvector);


    return optimal_threads;
}

int main() {
    FILE *file = fopen("colab_dataset.csv", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }


    fprintf(file, "optimalThreads,typeOp,matrixSize,numOp,numVar,varType,exeTime,isIterative,memoryPattern,complexity,matrixType\n");


    for (int matrixType = 0; matrixType <= 2; matrixType++) {
        printf("Testing with matrix type: %d\n", matrixType);


        for (int N = 100; N <= MAX_N; N += STEP_N) {
            printf("Processing matrix size: %d x %d\n", N, N);


            int **A = allocateMatrix(N);
            int **B = allocateMatrix(N);
            int **C = allocateMatrix(N);
            int **T = allocateMatrix(N);
            double **D = (double**)malloc(N * sizeof(double*));
            for (int i = 0; i < N; i++) {
                D[i] = (double*)malloc(N * sizeof(double));
            }


            generateMatrix(N, A, matrixType);
            generateMatrix(N, B, matrixType);




            printf("  Finding optimal thread count for multiplication...\n");
            int opt_threads_mul = findOptimalThreadCount(N, A, B, C, T, "Multiplication");
            double start = omp_get_wtime();
            matrixMultiply(N, A, B, C, opt_threads_mul);
            double end = omp_get_wtime();
            double time_taken = end - start;
            fprintf(file, "%d,Multiplication,%d,%d,3,int,%f,%d,%d,%d,%d\n",
                opt_threads_mul, N, N*N*N, time_taken, 0, 0, 3, matrixType);


            printf("  Finding optimal thread count for addition...\n");
            int opt_threads_add = findOptimalThreadCount(N, A, B, C, T, "Addition");
            start = omp_get_wtime();
            matrixAdd(N, A, B, C, opt_threads_add);
            end = omp_get_wtime();
            time_taken = end - start;
            fprintf(file, "%d,Addition,%d,%d,3,int,%f,%d,%d,%d,%d\n",
                opt_threads_add, N, N*N, time_taken, 0, 0, 2, matrixType);


            printf("  Finding optimal thread count for transposition...\n");
            int opt_threads_trans = findOptimalThreadCount(N, A, B, C, T, "Transposition");
            start = omp_get_wtime();
            matrixTranspose(N, A, T, opt_threads_trans);
            end = omp_get_wtime();
            time_taken = end - start;
            fprintf(file, "%d,Transposition,%d,%d,2,int,%f,%d,%d,%d,%d\n",
                opt_threads_trans, N, N*N, time_taken, 0, 1, 2, matrixType);


            printf("  Finding optimal thread count for determinant...\n");
            int opt_threads_det = findOptimalThreadCount(N, A, B, C, T, "Determinant");
            start = omp_get_wtime();
            double det = determinantSmart(N, A, opt_threads_det);
            end = omp_get_wtime();
            time_taken = end - start;
            fprintf(file, "%d,Determinant,%d,%d,1,int,%f,%d,%d,%d,%d\n",
                opt_threads_det, N, N*N*N, time_taken, 0, 0, 3, matrixType);
            printf("  Determinant of a %dx%d matrix calculated\n", N, N);




            printf("  Finding optimal thread count for scaling...\n");
            int opt_threads_scale = findOptimalThreadCount(N, A, B, C, T, "Scaling");
            start = omp_get_wtime();
            matrixScale(N, A, 2, C, opt_threads_scale);
            end = omp_get_wtime();
            time_taken = end - start;
            fprintf(file, "%d,Scaling,%d,%d,2,int,%f,%d,%d,%d,%d\n",
                opt_threads_scale, N, N*N, time_taken, 0, 1, 2, matrixType);


            printf("  Finding optimal thread count for exponential...\n");
            int opt_threads_exp = findOptimalThreadCount(N, A, B, C, T, "Exponential");
            start = omp_get_wtime();
            matrixExponential(N, A, D, opt_threads_exp);
            end = omp_get_wtime();
            time_taken = end - start;
            fprintf(file, "%d,Exponential,%d,%d,2,mixed,%f,%d,%d,%d,%d\n",
                opt_threads_exp, N, N*N, time_taken, 0, 1, 2, matrixType);


            printf("  Finding optimal thread count for logarithm...\n");
            int opt_threads_log = findOptimalThreadCount(N, A, B, C, T, "Logarithm");
            start = omp_get_wtime();
            matrixLogarithm(N, A, D, opt_threads_log);
            end = omp_get_wtime();
            time_taken = end - start;
            fprintf(file, "%d,Logarithm,%d,%d,2,mixed,%f,%d,%d,%d,%d\n",
                opt_threads_log, N, N*N, time_taken, 0, 1, 2, matrixType);


            printf("  Finding optimal thread count for square root...\n");
            int opt_threads_sqrt = findOptimalThreadCount(N, A, B, C, T, "SquareRoot");
            start = omp_get_wtime();
            matrixSqrt(N, A, D, opt_threads_sqrt);
            end = omp_get_wtime();
            time_taken = end - start;
            fprintf(file, "%d,SquareRoot,%d,%d,2,mixed,%f,%d,%d,%d,%d\n",
                opt_threads_sqrt, N, N*N, time_taken, 0, 1, 2, matrixType);



            printf("  Finding optimal thread count for LU decomposition...\n");
            int opt_threads_lu = findOptimalThreadCount(N, A, B, C, T, "LUDecomposition");
            start = omp_get_wtime();
            int **L = allocateMatrix(N);
            int **U = allocateMatrix(N);
            int *P = (int*)malloc(N * sizeof(int));
            luDecomposition(N, A, L, U, P, opt_threads_lu);
            end = omp_get_wtime();
            time_taken = end - start;
            fprintf(file, "%d,LUDecomposition,%d,%d,3,int,%f,%d,%d,%d,%d\n",
                opt_threads_lu, N, N*N*N/3, time_taken, 0, 0, 3, matrixType);
            freeMatrix(L, N);
            freeMatrix(U, N);
            free(P);


            printf("  Finding optimal thread count for eigenvalue calculation...\n");
            int opt_threads_eigen = findOptimalThreadCount(N, A, B, C, T, "Eigenvalue");
            start = omp_get_wtime();
            double eigenval;
            double *eigenvec = (double*)malloc(N * sizeof(double));
            powerMethod(N, A, &eigenval, &eigenvec, 100, 1e-6, opt_threads_eigen);
            end = omp_get_wtime();
            time_taken = end - start;
            fprintf(file, "%d,Eigenvalue,%d,%d,2,mixed,%f,%d,%d,%d,%d\n",
                opt_threads_eigen, N, N*N*100, time_taken, 1, 1, 2, matrixType);
            free(eigenvec);


            freeMatrix(A, N);
            freeMatrix(B, N);
            freeMatrix(C, N);
            freeMatrix(T, N);
            for (int i = 0; i < N; i++) {
                free(D[i]);
            }
            free(D);

            printf("Completed matrix size: %d x %d\n", N, N);
        }
    }

    fclose(file);
    printf("Enhanced dataset with optimal thread counts saved to enhanced_dataset.csv!\n");
    return 0;
}
