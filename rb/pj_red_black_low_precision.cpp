#include<iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include<fstream>
#include<iomanip>
#include <omp.h>
#define BLOCK_SIZE 16

using namespace std;

// Maximum number of iterations
#define ITER_MAX 100000000

// How often to check the relative residual
#define RESID_FREQ 1000 

// The residual
#define RESID 1e-6

float magnitude(float** T,int N);

int main(int argc, char** argv) {
    for (int N = 512; N < 540; N = N * 2) {
        int i, totiter;
        int done = 0;
        float** T = new float*[N];
        float** Ttmp = new float*[N];
        float** b = new float*[N];
        float bmag = 0;
        float resmag = 0;
        float m2 = 0.0001;
        float scale = 1.0 / (4.0 + m2);
        for (i = 0; i < N; i++) {
            T[i] = new float[N];
            Ttmp[i] = new float[N];
            b[i] = new float[N];
        }
        for (i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                T[i][j] = 0.0f;
                Ttmp[i][j] = 0.0f;
                b[i][j] = 0.0f;
            }
        }

        b[N / 2][N / 2] = 1;
        bmag = magnitude(b, N);
        printf("N = %d\n", N);
        printf("bmag: %.8e\n", bmag);
        cout << "magnitude = " << bmag << endl;

        std::chrono::time_point<std::chrono::steady_clock> begin_time = std::chrono::steady_clock::now();

        #pragma acc data copy(T[0:N][0:N]), copyin(b[0:N][0:N])
        {
            int left, right, up, down;
            for (totiter = RESID_FREQ; totiter < ITER_MAX && done == 0; totiter += RESID_FREQ) {
                for (int iter = 0; iter < RESID_FREQ; iter++) {
                    #pragma acc parallel loop 
                    for (int index = 0; index < N * N; index++) {
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
                    for (int index = 0; index < N * N; index++) {
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
                int i,j;
                float localres;

                i = 0;j = 0;
                localres = 0.0f;
                resmag = 0.0f;

                #pragma acc parallel loop collapse(2)
                for (i=0;i<N;i++) {
                    for (j=0;j<N;j++) {
                        left = (i - 1 + N) % N;  
                        right = (i + 1) % N;   
                        up = (j - 1 + N) % N;   
                        down = (j + 1) % N;     
                        localres = (b[i][j] - T[i][j] + scale*(T[i][up] + T[i][down] + T[left][j] + T[right][j]));
                        localres = localres*localres;
                        resmag = resmag + localres;
                    }
                }

                resmag = sqrt(resmag);

                printf("%d res %.8e bmag %.8e rel %.8e\n", totiter, resmag, bmag, resmag / bmag);
                if (resmag / bmag < RESID) {
                    done = 1;
                }
            }
        }

        std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> difference_in_time = end_time - begin_time;
        float difference_in_seconds = difference_in_time.count();
        printf("%d, %.8e\n", N, difference_in_seconds);
    }
    return 0;
}

float magnitude(float** T,int N)
{
   int i,j;
   float bmag;
   bmag = 0.0f;  
   for (i = 1; i<N; i++)
    for (j = 1; j<N; j++)
      bmag = bmag + T[i][j]*T[i][j];

   return sqrt(bmag);
}
