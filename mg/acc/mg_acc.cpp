#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <chrono>


typedef struct
{
    int N;
    int Lmax;
    int size[20];
    double a[20];
    double m2; // store m^2 directly
    double scale[20];
} param_t;

void relax(double *phi, double *res, int lev, int niter, param_t p);
void proj_res(double *res_c, double *rec_f, double *phi_f, int lev, param_t p);
void inter_add(double *phi_f, double *phi_c, int lev, param_t p);
double GetResRoot(double *phi, double *res, int lev, param_t p);

int main()
{
    #pragma acc init
    double *phi[20], *res[20];
    param_t p;
    int nlev;
    int i, lev;

    // set parameters________________________________________
    p.Lmax = 8; //5~9                    // max number of levels
    p.N = 2 * (int)pow(2, p.Lmax); // MUST BE POWER OF 2
    printf("N: %d\n", p.N);
    // set m^2 directly
    p.m2 = 0.0001;

    //nlev = 6; // NUMBER OF LEVELS:  nlev = 0 give top level alone
    nlev = p.Lmax - 1;
    if (nlev > p.Lmax)
    {
        printf("ERROR More levels than available in lattice! \n");
        return 0;
    }

    printf("\n V cycle for %d by %d lattice with nlev = %d out of max  %d \n", p.N, p.N, nlev, p.Lmax);

    // initialize arrays__________________________________
    p.size[0] = p.N;
    p.a[0] = 1.0;
    p.scale[0] = 1.0 / (4.0 + p.m2);

    for (lev = 1; lev < p.Lmax + 1; lev++)
    {
        p.size[lev] = p.size[lev - 1] / 2;
        p.a[lev] = 2.0 * p.a[lev - 1];
        // p.scale[lev] = 1.0/(4.0 + p.m*p.m*p.a[lev]*p.a[lev]);
        p.scale[lev] = 1.0 / (4.0 + p.m2);
    }

    for (lev = 0; lev < p.Lmax + 1; lev++)
    {
        phi[lev] = (double *)malloc(p.size[lev] * p.size[lev] * sizeof(double));

        res[lev] = (double *)malloc(p.size[lev] * p.size[lev] * sizeof(double));

        for (i = 0; i < p.size[lev] * p.size[lev]; i++)
        {
            phi[lev][i] = 0.0;
            res[lev][i] = 0.0;
        };
    }

    res[0][p.N / 2 + (p.N / 2) * p.N] = 1.0 * p.scale[0]; // unit point source in middle of N by N lattice

    // iterate to solve_____________________________________
    double resmag = 1.0; // not rescaled.
    int ncycle = 7;
    int n_per_lev = 10;
    resmag = GetResRoot(phi[0], res[0], 0, p);
    printf("At the %d cycle the mag residue is %g \n", ncycle, resmag);

    // while(resmag > 0.00001 && ncycle < 10000)
    #pragma acc data copy(phi[0:nlev][0:p.size[lev]*p.size[lev]], res[0:nlev][0:p.size[lev]*p.size[lev]])
    auto start = std::chrono::high_resolution_clock::now();
    while (resmag > 1e-6)
    {
        ncycle += 1;
        //if(ncycle ==10)break;
        for (lev = 0; lev < nlev; lev++) // go down
        {
            relax(phi[lev], res[lev], lev, n_per_lev, p);       // lev = 1, ..., nlev-1
            proj_res(res[lev + 1], res[lev], phi[lev], lev, p); // res[lev+1] += P^dag res[lev]
        }

        for (lev = nlev; lev >= 0; lev--) // come up
        {

            relax(phi[lev], res[lev], lev, n_per_lev, p); // lev = nlev -1, ... 0;
            if (lev > 0)
                inter_add(phi[lev - 1], phi[lev], lev, p); // phi[lev-1] += error = P phi[lev] and set phi[lev] = 0;
        }
        resmag = GetResRoot(phi[0], res[0], 0, p);
        printf("At the %d cycle the mag residue is %g \n", ncycle, resmag);
    }
    auto end = std::chrono::high_resolution_clock::now();  // 结束计时
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();  // 计算持续时间

    printf("The while loop took %ld microseconds.\n", duration);
    return 0;
}

void relax(double *phi, double *res, int lev, int niter, param_t p)
{
    int L;
    L = p.size[lev];
    double *phi_new = (double *)malloc(L * L * sizeof(double));
    
    #pragma acc kernels copy(phi[0:L*L], phi_new[0:L*L]) copyin(res[0:L*L])
    for (int i = 0; i < niter; i++)
    {
        #pragma acc loop independent        
        for (int x = 0; x < L; x++)
        {
            #pragma acc loop independent
            for (int y = 0; y < L; y++)
            {
                double phi_left = phi[(x - 1 + L) % L + y * L];
                double phi_right = phi[(x + 1) % L + y * L];
                double phi_up = phi[x + ((y - 1 + L) % L) * L];
                double phi_down = phi[x + ((y + 1) % L) * L];

                phi_new[x + y * L] = res[x + y * L] + p.scale[lev] * (phi_left + phi_right + phi_up + phi_down);
            }
        }

        #pragma acc loop independent
        for (int x = 0; x < L; x++)
        {
            #pragma acc loop independent
            for (int y = 0; y < L; y++)
            {
                phi[x + y * L] = phi_new[x + y * L];
            }
        }
    }
    free(phi_new);
    return;
}


void proj_res(double *res_c, double *res_f, double *phi_f, int lev, param_t p)
{
    int L, Lc, x, y;
    L = p.size[lev];
    double r[L * L];      // temp residue
    Lc = p.size[lev + 1]; // course level

    // get residue
    #pragma acc parallel loop present(res_f[0:L*L], phi_f[0:L*L], r[0:L*L])
    for (x = 0; x < L; x++)
        for (y = 0; y < L; y++)
            r[x + y * L] = res_f[x + y * L] - phi_f[x + y * L] + p.scale[lev] * (phi_f[(x + 1) % L + y * L] + phi_f[(x - 1 + L) % L + y * L] + phi_f[x + ((y + 1) % L) * L] + phi_f[x + ((y - 1 + L) % L) * L]);

    // project residue
    #pragma acc parallel loop present(r[0:L*L], res_c[0:Lc*Lc])
    for (x = 0; x < Lc; x++)
        for (y = 0; y < Lc; y++)
            res_c[x + y * Lc] = 0.25 * (r[2 * x + 2 * y * L] + r[(2 * x + 1) % L + 2 * y * L] + r[2 * x + ((2 * y + 1)) % L * L] + r[(2 * x + 1) % L + ((2 * y + 1) % L) * L]);
    return;
}

void inter_add(double *phi_f, double *phi_c, int lev, param_t p)
{
    int L, Lc, x, y;
    Lc = p.size[lev]; // coarse  level
    L = p.size[lev - 1];

    #pragma acc parallel loop present(phi_f[0:L*L], phi_c[0:Lc*Lc])
    for (x = 0; x < Lc; x++)
        for (y = 0; y < Lc; y++)
        {
            phi_f[2 * x + 2 * y * L] += phi_c[x + y * Lc];
            phi_f[(2 * x + 1) % L + 2 * y * L] += phi_c[x + y * Lc];
            phi_f[2 * x + ((2 * y + 1)) % L * L] += phi_c[x + y * Lc];
            phi_f[(2 * x + 1) % L + ((2 * y + 1) % L) * L] += phi_c[x + y * Lc];
        }
    
    // set to zero so phi = error
    #pragma acc parallel loop present(phi_c[0:Lc*Lc])
    for (x = 0; x < Lc; x++)
        for (y = 0; y < Lc; y++){
            phi_c[x + y * Lc] = 0.0;
        }

    return;
}

double GetResRoot(double *phi, double *res, int lev, param_t p)
{ // true residue
    int x, y;
    double residue;
    double ResRoot = 0.0;
    int L;
    L = p.size[lev];
    #pragma acc parallel loop reduction(+:ResRoot) present(phi[0:L*L], res[0:L*L])
    for (x = 0; x < L; x++)
        for (y = 0; y < L; y++)
        {
            residue = res[x + y * L] / p.scale[lev] - phi[x + y * L] / p.scale[lev] + (phi[(x + 1) % L + y * L] + phi[(x - 1 + L) % L + y * L] + phi[x + ((y + 1) % L) * L] + phi[x + ((y - 1 + L) % L) * L]);
            ResRoot += residue * residue; // true residue
        }

    return sqrt(ResRoot);
}
