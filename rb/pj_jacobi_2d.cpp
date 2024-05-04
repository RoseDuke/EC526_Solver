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
void jacobi(double** T, double** b, double** tmp,int N, double scale);
double getResid(double** T, double** b,int N, double scale);
void redBlackGaussSeidel(double** T, double** b, int N, double scale);
void redBlackGaussSeidelOpenACC(double** T, double** b, int N, double scale);
void redBlackGaussSeidelOpenACC_collapse(double** T, double** b, int N, double scale);
void redBlackGaussSeidelBlock(double** T, double** b, int N, double scale);

int main(int argc, char** argv) {
    for (int N = 32; N < 40; N = N * 2) {
        int i, totiter;
        int done = 0;
        double** T = new double*[N];
        double** Ttmp = new double*[N];
        double** b = new double*[N];
        double bmag = 0;
        double resmag = 0;
        double m2 = 0.0001;
        double scale = 1.0 / (4.0 + m2);
        printf("scale: %f\n", scale);
        for (i = 0; i < N; i++) {
            T[i] = new double[N];
            Ttmp[i] = new double[N];
            b[i] = new double[N];
        }
        for (i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                T[i][j] = 0.0;
                Ttmp[i][j] = 0.0;
                b[i][j] = 0.0;
            }
        }

        b[N / 2][N / 2] = 1;
        bmag = magnitude(b, N);
        printf("N = %d\n", N);
        printf("bmag: %.8e\n", bmag);
        cout << "magnitude = " << bmag << endl;
        std::chrono::time_point<std::chrono::steady_clock> begin_time = std::chrono::steady_clock::now();
        for (totiter = RESID_FREQ; totiter < ITER_MAX && done == 0; totiter += RESID_FREQ) {
            jacobi(T, b, Ttmp, N, scale);
            // redBlackGaussSeidel(T, b, N, scale);
            // redBlackGaussSeidelOpenACC(T, b, N, scale);
            // redBlackGaussSeidelOpenACC_collapse(T, b, N, scale);
            // redBlackGaussSeidelBlock(T, b, N, scale);

            resmag = getResid(T, b, N, scale);

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
   for (i = 0; i<N; i++)
    for (j = 0; j<N; j++)
      bmag = bmag + T[i][j]*T[i][j];

   return sqrt(bmag);
}

void jacobi(double** T, double** b, double** tmp, int N, double scale) {
    int iter, i, j;
    int left, right, up, down;

    for (iter = 0; iter < RESID_FREQ; iter++) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                left = (i - 1 + N) % N;  
                right = (i + 1) % N;   
                up = (j - 1 + N) % N;   
                down = (j + 1) % N;  
                // if (iter == 1) {
                //     printf("left: %d\n", left);
                //     printf("right: %d\n", right);
                //     printf("up: %d\n", up);
                //     printf("down: %d\n", down);
                // }   
                
                tmp[i][j] = scale * (T[i][up] + T[i][down] + T[left][j] + T[right][j]) + b[i][j];
                // if (i==3 && j == 4) {
                //     printf("left, j: %d, %d\n", left, j);
                //     printf("right, j: %d, %d\n", right, j);
                //     printf("i, up: %d, %d\n", i, up);
                //     printf("i, down: %d, %d\n", i, down);
                //     printf("up, down, left, right: %f, %f, %f, %f\n", T[i][up], T[i][down], T[left][j], T[right][j]);
                //     printf("tmp_ij: %f\n", tmp[i][j]);
                //     printf("scale: %f\n", scale);
                // }
                // if (tmp[i][j] != 0) {
                //     printf("i, j: %d, %d\n", i, j);
                //     printf("tmp_ij: %f\n", tmp[i][j]);
                // }
            }
        }

        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                T[i][j] = tmp[i][j];
                // if (T[i][j] != 0) {
                //     printf("i, j: %d, %d\n", i, j);
                //     printf("T_ij: %f\n", T[i][j]);
                // }
            }
        }
    }
}


void redBlackGaussSeidel(double** T, double** b, int N, double scale) {
    int iter, i, j;
    iter = 0; i = 0; j = 0;
    int left, right, up, down;

    for (iter = 0; iter < RESID_FREQ; iter++) {
        for (i = 0; i < N; i++) {
            for (j = (i % 2); j < N; j += 2) {
                left = (i - 1 + N) % N;  
                right = (i + 1) % N;   
                up = (j - 1 + N) % N;   
                down = (j + 1) % N;  
                T[i][j] = scale * (T[i][up] + T[i][down] + T[left][j] + T[right][j]) + b[i][j];
            }
        }
 
        for (i = 0; i < N; i++) {
            for (j = ((i + 1) % 2); j < N; j += 2) {
                left = (i - 1 + N) % N;  
                right = (i + 1) % N;   
                up = (j - 1 + N) % N;   
                down = (j + 1) % N;  
                T[i][j] = scale * (T[i][up] + T[i][down] + T[left][j] + T[right][j]) + b[i][j];
            }
        }
    }
}

void redBlackGaussSeidelOpenACC(double** T, double** b, int N, double scale) {
    int iter, i, j;
    iter = 0; i = 0; j = 0;
    int left, right, up, down;

    #pragma acc data copy(T[0:N][0:N]), copy(b[0:N][0:N])
    {
        for (iter = 0; iter < RESID_FREQ; iter++) {
            #pragma acc parallel loop 
            for (i = 0; i < N; i++) {
                for (j = (i % 2); j < N; j += 2) {
                    left = (i - 1 + N) % N;  
                    right = (i + 1) % N;   
                    up = (j - 1 + N) % N;   
                    down = (j + 1) % N;  
                    T[i][j] = scale * (T[i][up] + T[i][down] + T[left][j] + T[right][j]) + b[i][j];
                }
            }

            #pragma acc parallel loop 
            for (i = 0; i < N; i++) {
                for (j = ((i + 1) % 2); j < N; j += 2) {
                    left = (i - 1 + N) % N;  
                    right = (i + 1) % N;   
                    up = (j - 1 + N) % N;   
                    down = (j + 1) % N;  
                    T[i][j] = scale * (T[i][up] + T[i][down] + T[left][j] + T[right][j]) + b[i][j];
                }
            }
        }
    }
}

void redBlackGaussSeidelOpenACC_collapse(double** T, double** b, int N, double scale) {
    int iter, index;
    int left, right, up, down;
    #pragma acc data copy(T[0:N][0:N]), copyin(b[0:N][0:N])
    {
        for (iter = 0; iter < RESID_FREQ; iter++) {
            #pragma acc parallel loop 
            for (index = 0; index < N * N; index++) {
                int i = index / N;
                int j = index % N;
                if ((i + j) % 2 == 0) { 
                    left = (i - 1 + N) % N;  
                    right = (i + 1) % N;   
                    up = (j - 1 + N) % N;   
                    down = (j + 1) % N;     
                    T[i][j] = scale * (T[i][up] + T[i][down] + T[left][j] + T[right][j]) + b[i][j];
                }
            }

            #pragma acc parallel loop 
            for (index = 0; index < N * N; index++) {
                int i = index / N;
                int j = index % N;
                if ((i + j) % 2 == 1) { 
                    left = (i - 1 + N) % N;  
                    right = (i + 1) % N;   
                    up = (j - 1 + N) % N;   
                    down = (j + 1) % N;      
                    T[i][j] = scale * (T[i][up] + T[i][down] + T[left][j] + T[right][j]) + b[i][j];
                }
            }
        }
    }
}

void redBlackGaussSeidelBlock(double** T, double** b, int N, double scale) {
    int iter, index, endIndex;
    int left, right, up, down;
    #pragma acc data copy(T[0:N][0:N]), copy(b[0:N][0:N])
    {
        for (iter = 0; iter < RESID_FREQ; iter++) {
            #pragma acc parallel loop 
            for (index = 0; index < N * N; index += BLOCK_SIZE) {
                endIndex = fmin(index + BLOCK_SIZE, N * N); 
                for (int idx = index; idx < endIndex; idx++) {
                    int i = idx / N;
                    int j = idx % N;
                    if ((i + j) % 2 == 0) {
                        left = (i - 1 + N) % N;  
                        right = (i + 1) % N;   
                        up = (j - 1 + N) % N;   
                        down = (j + 1) % N;     
                        T[i][j] = scale * (T[i][up] + T[i][down] + T[left][j] + T[right][j]) + b[i][j];
                    }
                }
            }

            #pragma acc parallel loop 
            for (index = 0; index < N * N; index += BLOCK_SIZE) {
                endIndex = fmin(index + BLOCK_SIZE, N * N); 
                for (int idx = index; idx < endIndex; idx++) {
                    int i = idx / N;
                    int j = idx % N;
                    if ((i + j) % 2 == 1) {
                        left = (i - 1 + N) % N;  
                        right = (i + 1) % N;   
                        up = (j - 1 + N) % N;   
                        down = (j + 1) % N;      
                        T[i][j] = scale * (T[i][up] + T[i][down] + T[left][j] + T[right][j]) + b[i][j];
                    }
                }
            }
        }
    }
}

double getResid(double** T, double** b,int N, double scale)
{
  int i,j;
  double localres,resmag;
  int left, right, up, down;
  i = 0;j = 0;
  localres = 0.0;
  resmag = 0.0;

  for (i=0;i<N;i++)
  for (j=0;j<N;j++)
  {
    left = (i - 1 + N) % N;  
    right = (i + 1) % N;   
    up = (j - 1 + N) % N;   
    down = (j + 1) % N;     
    // printf("b: %f\n", b[i][j]);
    localres = (b[i][j] - T[i][j] + scale*(T[i][up] + T[i][down] + T[left][j] + T[right][j]));
    // if (b[i][j] > 0) {
    //     printf("Tij: %f\n", T[i][j]);
    //     printf("localres: %f\n", localres);
    // }
    
    localres = localres*localres;
    resmag = resmag + localres;
  }
//   printf("resmag: %f\n", resmag);
  resmag = sqrt(resmag);

  return resmag;
}
