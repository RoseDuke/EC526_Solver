#include <cmath>
#include <iostream>
#include <fstream>
#include <complex.h>
#include <openacc.h>
#include <chrono>

#define PI 3.141592653589793
#define I Complex(0.0, 1.0)
typedef std::complex<double> Complex;
using namespace std::chrono;

// #define L 8           // grid size
#define R 0.5         // relaxation factor
#define MAX_ITER 1000 // maximum number of iterations
// #define m_squared 0.1

void FFTrecursion(Complex *Fout, Complex *Fin, Complex *omega, int N, int Nfft)
{

    // cout << "Nfft =  " << Nfft << endl;
    // cout << "Enters at  N = " << N << endl;
    if (N == 2) // N=2, calculate the butterfly operation and exit
    {
        Fout[0] = Fin[0] + Fin[1];
        Fout[1] = Fin[0] - Fin[1];
        return;
    }

    Complex *Fin_even = new Complex[N / 2];
    Complex *Fout_even = new Complex[N / 2];
    Complex *Fin_odd = new Complex[N / 2];
    Complex *Fout_odd = new Complex[N / 2];

    for (int i = 0; i < N / 2; i++)
    {
        Fin_even[i] = Fin[2 * i];
        Fin_odd[i] = Fin[2 * i + 1];
    }

    // cout << "Calls at twice at N =  " << N / 2 << endl;
    FFTrecursion(Fout_even, Fin_even, omega, N / 2, Nfft); // recursively do the evern and odd part
    FFTrecursion(Fout_odd, Fin_odd, omega, N / 2, Nfft);

    int lev = Nfft / N;
    // cout << " at level =  " << lev << " Combine two  N =  " << N / 2 << endl;

    for (int i = 0; i < N / 2; i++)
    {                                                          // N = 4  need omega_4(i) = (omega_16)^4
        Fout[i] = Fout_even[i] + omega[i * lev] * Fout_odd[i]; // combine the even and odd part bu butterfly operation
        Fout[i + N / 2] = Fout_even[i] - omega[i * lev] * Fout_odd[i];
    }
}

void FFTrecursion_inv(Complex *Fout, Complex *Fin, Complex *omega, int N, int Nfft)
{
    // cout << "Inters at  N " << N << endl;
    if (N == 2)
    {
        Fout[0] = (Fin[0] + Fin[1]) / (double)Nfft;
        Fout[1] = (Fin[0] - Fin[1]) / (double)Nfft;
        return;
    }

    Complex *Fin_even = new Complex[N / 2];
    Complex *Fout_even = new Complex[N / 2];
    Complex *Fin_odd = new Complex[N / 2];
    Complex *Fout_odd = new Complex[N / 2];

    for (int i = 0; i < N / 2; i++)
    {
        Fin_even[i] = Fin[2 * i];
        Fin_odd[i] = Fin[2 * i + 1];
    }

    // cout << "Calls at twice at N =  "<< N/2 << endl;
    FFTrecursion_inv(Fout_even, Fin_even, omega, N / 2, Nfft);
    FFTrecursion_inv(Fout_odd, Fin_odd, omega, N / 2, Nfft);

    int lev = Nfft / N;
    // cout << " Combine two  N =  " << N/2 << " at level =  " << lev << endl;

    for (int i = 0; i < N / 2; i++)
    {
        Fout[i] = Fout_even[i] + conj(omega[i * lev]) * Fout_odd[i];
        Fout[i + N / 2] = Fout_even[i] - conj(omega[i * lev]) * Fout_odd[i];
    }
}

int main()
{
    int L; // grid size
    // std::cin >> L;
    int i, j, iter;

    std::ofstream file("output.dat");
    if (!file)
    {
        std::cerr << "Error: file could not be opened" << std::endl;
        return 1;
    }
    for (int power = 20; power <= 36; power++)
    {
        for (int m = 0; m <= 5; m++)
        {
            double m_squared = 1.0 / (double)pow(10.0, (double)m); // msquared=1/10^m

            double ex = power / 4.0;
            L = (int)(pow)(2, ex);
            int N = (int)pow(2, ceil(log2(L))); // N=2^ceil(log2(L)), nearest 2^k of L
            int extended_size = 2 * N;          /// Actual size of the grid=2*N

            // allocate memory
            Complex omega[extended_size];
            Complex *px = new Complex[extended_size * extended_size]();
            Complex *px_new = new Complex[extended_size * extended_size]();
            // px = (Complex *)malloc(extended_size * extended_size * sizeof(Complex));
            // px_new = (Complex *)malloc(extended_size * extended_size * sizeof(Complex));

            // initialize grid of L*L
            for (i = 0; i < L; i++)
            {
                for (j = 0; j < L; j++)
                {
                    px[i * L + j] = 0.0 + 0.0 * I;
                }
            }
            px[L / 2 * L + L / 2] = 1.0 + 0.0 * I; // central point

            // Time start
            auto start = high_resolution_clock::now();

            // extend the grid to extended_size*extended_size
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    Complex val = px[i * extended_size + j];
                    px[(i + N) * extended_size + j] = -val;      // copy inverted to bottom
                    px[i * extended_size + (j + N)] = -val;      // copy inverted to right
                    px[(i + N) * extended_size + (j + N)] = val; // copy original to bottom-right
                }
            }

            // change the size of actual grid , together with Nfft = L
            L = extended_size;
            int Nfft = L; // nfft=L

            // initialize FFT omega array
            for (i = 0; i < L; i++)
            {
                omega[i] = exp(-2 * M_PI * i / L * I);
            }

            // perform FFT on initial grid
            FFTrecursion(px, px, omega, L, Nfft);

            // iterate using OpenACC
            // #pragma acc data copy(px) copy(px_new)
#pragma acc data copy(px[0 : extended_size * extended_size]), create(px_new[0 : extended_size * extended_size])
            for (iter = 0; iter < MAX_ITER; iter++)
            {
#pragma acc kernels loop independent
                for (i = 0; i < L; i++)
                {
                    for (j = 0; j < L; j++)
                    {
                        int idx = i * L + j;
                        px_new[idx] = (1 - R) * px[idx] + R / (4 + m_squared) * (px[((i + 1) % L) * L + j] + px[((i - 1 + L) % L) * L + j] + px[i * L + ((j + 1) % L)] + px[i * L + ((j - 1 + L) % L)]);
                    }
                }

                // swap pointers
                Complex *tmp = px;
                px = px_new;
                px_new = tmp;
            }

            // perform inverse FFT
            FFTrecursion_inv(px, px, omega, L, Nfft);

            // Time Stop
            auto stop = high_resolution_clock::now();

            // Dudation
            auto duration = duration_cast<microseconds>(stop - start);

            // Print time
            std::cout << "For L= " << (int)pow(2, ex) << " and m^2 =" << m_squared << " , Time taken by function is: " << duration.count() << " *10^-6 seconds" << std::endl;

            // print to the data file
            file << (int)pow(2, ex) << "\t" << m_squared << "\t" << duration.count() << std::endl;

            // free memory
            // free(px);
            // free(px_new);

            delete[] px;
            delete[] px_new;
        }
    }
    return 0;
}
