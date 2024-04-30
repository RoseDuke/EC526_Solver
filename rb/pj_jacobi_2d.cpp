#include<iostream>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include<fstream>
#include<iomanip>
#include <omp.h>
#define BLOCK_SIZE 16

using namespace std;

// 1D length
//#define N 512

// Maximum number of iterations
#define ITER_MAX 100000000

// How often to check the relative residual
#define RESID_FREQ 1000 

// The residual
#define RESID 1e-6

double magnitude(double** T,int N);
void jacobi(double** T, double** b, double** tmp,int N);
double getResid(double** T, double** b,int N);
void redBlackGaussSeidel(double** T, double** b, int N);
void redBlackGaussSeidelOpenACC(double** T, double** b, int N);
void redBlackGaussSeidelOpenACC_collapse(double** T, double** b, int N);
void redBlackGaussSeidelBlock(double** T, double** b, int N);

int main(int argc, char** argv) {
    // ofstream f;
    // f.open("jacobi_scal_2d_res.csv");
    for (int N = 1024; N < 1040; N = N * 2) {
        int i, totiter;
        int done = 0;
        double** T = new double*[N + 2];
        double** Ttmp = new double*[N + 2];
        double** b = new double*[N + 2];
        double bmag = 0;
        double resmag = 0;
        for (i = 0; i < N + 2; i++) {
            T[i] = new double[N + 2];
            Ttmp[i] = new double[N + 2];
            b[i] = new double[N + 2];
        }
        for (i = 0; i < N + 2; i++) {
            for (int j = 0; j < N + 2; j++) {
                T[i][j] = 0.0;
                Ttmp[i][j] = 0.0;
                b[i][j] = 0.0;
            }
        }

        b[N / 2][N / 2] = 100.0;
        bmag = magnitude(b, N);
        printf("N = %d\n", N);
        printf("bmag: %.8e\n", bmag);
        cout << "magnitude = " << bmag << endl;

        std::chrono::time_point<std::chrono::steady_clock> begin_time = std::chrono::steady_clock::now();

        for (totiter = RESID_FREQ; totiter < ITER_MAX && done == 0; totiter += RESID_FREQ) {
            // jacobi(T, b, Ttmp, N);
            // redBlackGaussSeidel(T, b, N);
            // redBlackGaussSeidelOpenACC(T, b, N);
            redBlackGaussSeidelOpenACC_collapse(T, b, N);
            // redBlackGaussSeidelBlock(T, b, N);

            resmag = getResid(T, b, N);

            printf("%d res %.8e bmag %.8e rel %.8e\n", totiter, resmag, bmag, resmag / bmag);
            if (resmag / bmag < RESID) {
                done = 1;
            }
        }

        std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> difference_in_time = end_time - begin_time;
        double difference_in_seconds = difference_in_time.count();

        printf("%d, %.8e\n", N, difference_in_seconds);
    }
    return 0;
}

double magnitude(double** T,int N)
{
   int i,j;
   double bmag;
   bmag = 0.0;  
   for (i = 1; i<N; i++)
    for (j = 1; j<N; j++)
      bmag = bmag + T[i][j]*T[i][j];

   return sqrt(bmag);
}

void jacobi(double** T, double** b, double** tmp, int N)
{
    int iter, i, j;
    iter = 0; i = 0; j = 0;

    for (iter = 0; iter < RESID_FREQ; iter++) {
        for (i = 1; i < N; i++) {
            for (j = 1; j < N; j++) {
                tmp[i][j] = 0.25 * (T[i][j + 1] + T[i][j - 1] + T[i + 1][j] + T[i - 1][j]) + b[i][j];
            }
        }

        for (i = 1; i < N; i++) {
            for (j = 1; j < N; j++) {
                T[i][j] = tmp[i][j];
            }
        }
    }
}

void redBlackGaussSeidel(double** T, double** b, int N) {
    int iter, i, j;
    iter = 0; i = 0; j = 0;

    for (iter = 0; iter < RESID_FREQ; iter++) {
        for (i = 1; i < N; i++) {
            for (j = 1 + (i % 2); j < N; j += 2) {
                T[i][j] = 0.25 * (T[i][j + 1] + T[i][j - 1] + T[i + 1][j] + T[i - 1][j]) + b[i][j];
            }
        }
 
        for (i = 1; i < N; i++) {
            for (j = 1 + ((i + 1) % 2); j < N; j += 2) {
                T[i][j] = 0.25 * (T[i][j + 1] + T[i][j - 1] + T[i + 1][j] + T[i - 1][j]) + b[i][j];
            }
        }
    }
}

void redBlackGaussSeidelOpenACC(double** T, double** b, int N) {
    int iter, i, j;
    iter = 0; i = 0; j = 0;

    #pragma acc data copy(T[0:N+2][0:N+2]), copy(b[0:N+2][0:N+2])
    {
        for (iter = 0; iter < RESID_FREQ; iter++) {
            #pragma acc parallel loop 
            for (i = 1; i < N; i++) {
                for (j = 1 + (i % 2); j < N; j += 2) {
                    T[i][j] = 0.25 * (T[i][j + 1] + T[i][j - 1] + T[i + 1][j] + T[i - 1][j]) + b[i][j];
                }
            }

            #pragma acc parallel loop 
            for (i = 1; i < N; i++) {
                for (j = 1 + ((i + 1) % 2); j < N; j += 2) {
                    T[i][j] = 0.25 * (T[i][j + 1] + T[i][j - 1] + T[i + 1][j] + T[i - 1][j]) + b[i][j];
                }
            }
        }
    }
}

void redBlackGaussSeidelOpenACC_collapse(double** T, double** b, int N) {
    int iter, index;
    
    #pragma acc data copy(T[0:N+2][0:N+2]), copyin(b[0:N+2][0:N+2])
    {
        for (iter = 0; iter < RESID_FREQ; iter++) {
            #pragma acc parallel loop 
            for (index = 1; index < N * N; index++) {
                int i = index / N;
                int j = index % N;
                if ((i + j) % 2 == 0 && i > 0 && j > 0 && i < N && j < N) { 
                    T[i][j] = 0.25 * (T[i][j + 1] + T[i][j - 1] + T[i + 1][j] + T[i - 1][j]) + b[i][j];
                }
            }

            #pragma acc parallel loop 
            for (index = 1; index < N * N; index++) {
                int i = index / N;
                int j = index % N;
                if ((i + j) % 2 == 1 && i > 0 && j > 0 && i < N && j < N) {  
                    T[i][j] = 0.25 * (T[i][j + 1] + T[i][j - 1] + T[i + 1][j] + T[i - 1][j]) + b[i][j];
                }
            }
        }
    }
}

void redBlackGaussSeidelBlock(double** T, double** b, int N) {
    int iter, index, endIndex;

    #pragma acc data copy(T[0:N+2][0:N+2]), copy(b[0:N+2][0:N+2])
    {
        for (iter = 0; iter < RESID_FREQ; iter++) {
            // 处理红色点
            #pragma acc parallel loop 
            for (index = 1; index < N * N; index += BLOCK_SIZE) {
                endIndex = fmin(index + BLOCK_SIZE, N * N); // 确保不越界
                for (int idx = index; idx < endIndex; idx++) {
                    int i = idx / N;
                    int j = idx % N;
                    if ((i + j) % 2 == 0 && i > 0 && j > 0 && i < N && j < N) {
                        T[i][j] = 0.25 * (T[i][j + 1] + T[i][j - 1] + T[i + 1][j] + T[i - 1][j]) + b[i][j];
                    }
                }
            }

            // 处理黑色点
            #pragma acc parallel loop 
            for (index = 1; index < N * N; index += BLOCK_SIZE) {
                endIndex = fmin(index + BLOCK_SIZE, N * N); // 确保不越界
                for (int idx = index; idx < endIndex; idx++) {
                    int i = idx / N;
                    int j = idx % N;
                    if ((i + j) % 2 == 1 && i > 0 && j > 0 && i < N && j < N) {
                        T[i][j] = 0.25 * (T[i][j + 1] + T[i][j - 1] + T[i + 1][j] + T[i - 1][j]) + b[i][j];
                    }
                }
            }
        }
    }
}

double getResid(double** T, double** b,int N)
{
  int i,j;
  double localres,resmag;

  i = 0;j = 0;
  localres = 0.0;
  resmag = 0.0;

  for (i=1;i<N;i++)
  for (j=1;j<N;j++)
  {
    localres = (b[i][j] - T[i][j] + 0.25*(T[i][j+1] + T[i][j-1] + T[i+1][j] + T[i-1][j]));
    localres = localres*localres;
    resmag = resmag + localres;
  }

  resmag = sqrt(resmag);

  return resmag;
}
