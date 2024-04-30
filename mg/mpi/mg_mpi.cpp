#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>
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

// Useful globals
int world_size; // number of processes
int my_rank; // my process number

void relax(double *phi, double *res, int lev, int niter, param_t p);
void proj_res(double *res_c, double *rec_f, double *phi_f, int lev, param_t p);
void inter_add(double *phi_f, double *phi_c, int lev, param_t p);
double GetResRoot(double *phi, double *res, int lev, param_t p);


void allgather(double * array, int len){
    int local_size = len / world_size;
    double *tmp = (double *)malloc(sizeof(double) * len);

    MPI_Allgather(array + my_rank * local_size, local_size, MPI_DOUBLE,
                  tmp, local_size, MPI_DOUBLE, MPI_COMM_WORLD);

    memcpy(array, tmp, len * sizeof(double));
    free(tmp);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double *phi[20], *res[20];
    param_t p;
    int nlev;
    int i, lev;

    // set parameters________________________________________
    // L = 2^(p.Lmax+1)
    p.Lmax = 4;                    // max number of levels, level = Lmax - 1, 4 = L32
    p.N = 2 * (int)pow(2, p.Lmax); // MUST BE POWER OF 2
    if(my_rank == 0)printf("N: %d\n", p.N);
    
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


    for(int k = 0; k < p.Lmax+1; k++){
        for(long long i = 0; i < p.size[k] * p.size[k]; i++){
            //phi[k][i] = i;
        }
    }
    
    /*
    printf("\n");
    for(int i = 0; i < p.size[2] * p.size[2]; i++){
        if(i% p.size[2] == 0) printf("\n");
        printf("%f ", phi[2][i]);
    }
    printf("\n");
    */

    res[0][p.N / 2 + (p.N / 2) * p.N] = 1.0 * p.scale[0]; // unit point source in middle of N by N lattice

    // iterate to solve_____________________________________
    double resmag = 1.0; // not rescaled.
    int ncycle = 7;
    int n_per_lev = 10;


    auto start = std::chrono::high_resolution_clock::now();
    resmag = GetResRoot(phi[0], res[0], 0, p);
    printf("At the %d cycle the mag residue is %g \n", ncycle, resmag);

    int iter = 0;
    while (resmag > 1e-6)
    {
        ncycle += 1;
        //if(ncycle == 10)break;
        for (lev = 0; lev < nlev; lev++) // go down
        {
            relax(phi[lev], res[lev], lev, n_per_lev, p);  

            allgather(phi[lev], p.size[lev] * p.size[lev]);

            proj_res(res[lev + 1], res[lev], phi[lev], lev, p); 
            
            // Bcast can be changed to send and recv.
            //MPI_Bcast(res[lev + 1], p.size[lev + 1] * p.size[lev + 1], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        
        for (lev = nlev; lev >= 0; lev--) // come up
        {
            relax(phi[lev], res[lev], lev, n_per_lev, p); 
            allgather(phi[lev], p.size[lev] * p.size[lev]);

            if(lev > 0){
                //if(my_rank == 0){
                    inter_add(phi[lev - 1], phi[lev], lev, p);

                //}
                //MPI_Bcast(phi[lev - 1], p.size[lev - 1] * p.size[lev - 1], MPI_DOUBLE, 0, MPI_COMM_WORLD);
                //memset(phi[lev], 0, sizeof(phi[lev]));
            }
        }

        
        resmag = GetResRoot(phi[0], res[0], 0, p);
        if(my_rank == 0)
            printf("At the %d cycle the mag residue is %g \n", ncycle, resmag);
        
    }
    auto end = std::chrono::high_resolution_clock::now();  
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); 
    if(my_rank == 0)printf("The while loop took %ld microseconds.\n", duration);    

    for (lev = 0; lev < p.Lmax + 1; lev++)
    {
        free(phi[lev]);
        free(res[lev]);
    }

    MPI_Finalize();

    return 0;
}

// mpirun -np 4 ./mg

/*
low_buffer[L]
phi[local_size][L]
high_buffer[L]
*/
void relax(double *phi, double *res, int lev, int niter, param_t p)
{
    int i, x, y;
    int L;
    L = p.size[lev];

    int local_size = L / world_size;
   //printf("local_size %d\n", local_size);
   MPI_Request request[4];
   int requests;
   MPI_Status status[4];

   //printf("relax line 163\n");

    double **phi_new;
    phi_new = (double**)malloc(local_size * sizeof(double*));
    for(int i = 0; i < local_size; i++) {
        phi_new[i] = (double*)malloc(L * sizeof(double));
    }

    double *high_buffer = (double*)malloc(L * sizeof(double));
    double *low_buffer = (double*)malloc(L * sizeof(double));


   //printf("relax line 184\n");
    for (int iter = 0; iter < niter; iter++){

        
        requests=0;
        
        // Fill the lower buffer. Send to the right, listen from the left.
        // send the last line to next rank 
        MPI_Isend(&(phi[((my_rank + 1) * local_size - 1) * L]),  L, MPI_DOUBLE, (my_rank+1)%world_size, 1, MPI_COMM_WORLD, request + requests++);
        MPI_Irecv(low_buffer, L, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 1, MPI_COMM_WORLD, request + requests++);
        
        // Fill the higher buffer. Send to the left, listen from the right.
        MPI_Isend(&phi[(my_rank * local_size) * L],   L, MPI_DOUBLE, (my_rank+world_size-1)%world_size, 0, MPI_COMM_WORLD, request + requests++);
        MPI_Irecv(high_buffer, L, MPI_DOUBLE, (my_rank+1)%world_size, 0, MPI_COMM_WORLD, request + requests++);
        //printf("I am rank %d of %d and I received %.8e from the right.\n", my_rank, world_size, right_buffer);
        
        /* debug log for MPI
        if(my_rank == 0){
            printf("low_buffer\n");
            
            for(int i = 0; i < L; i++){
                printf("%.1f ", low_buffer[i]);
            }

            printf("\n");
            printf("high_buffer\n");
            for(int i = 0; i < L; i++){
                printf("%.1f ", high_buffer[i]);
            }
            printf("\n");
        }
        */
        //exit(0);
        

        int start_line = my_rank*local_size;
        int end_line = (my_rank + 1)*local_size - 1;

        // [0, local_size - 1]
        // this branch handle => [1, local_size - 2]
        for (int x = 1; x < local_size - 1; x++)
        {   
            for (int y = 0; y < L; y++)
            {
                //if(iter==0)printf("%.1f ", res[x*L + y]);
                double left = phi[(start_line+x)*L + (y-1+L)%L];
                double right = phi[(start_line+x)*L + (y+1+L)%L];
                double low = phi[(start_line+x-1)*L + y];
                double high = phi[(start_line+x+1)*L + y];

                phi_new[x][y] = res[(start_line+x)*L + y] + p.scale[lev] * (left + right + low + high);

                //if(iter==0)printf("%.1f ", phi_new[x][y]);
            }
            //if(iter==0)printf("\n");
            
        }

        MPI_Waitall(requests, request, status );

        // phi_new[0, local_size - 1] => res[(my_rank*local_size) * L + y] ~ res[((my_rank+1)*local_size -1) * L + y]
        // this branch handle => line:0 
        for(int y = 0; y < L; y++){
            double left = phi[start_line*L + (y-1+L)%L];
            double right = phi[start_line*L + (y+1+L)%L];
            double low = low_buffer[y];
            double high = phi[(start_line+1)*L + y];    
            phi_new[0][y] = res[start_line * L + y] + p.scale[lev] * (left + right + low + high);
            //if(res[0*L + y] !=0)printf("6666666\n");
        }
        // this branch handle => line:local_size - 1
        for(int y = 0; y < L; y++){
            double left = phi[end_line*L + (y-1+L)%L];
            double right = phi[end_line*L + (y+1+L)%L];
            double low = phi[(end_line - 1)*L + y];
            double high = high_buffer[y];    
            phi_new[local_size - 1][y] = res[end_line * L + y] + p.scale[lev] * (left + right + low + high);
        }
        
        for (int x = 0; x < local_size; x++)
        {
            for (int y = 0; y < L; y++)
            {
                phi[(start_line+x) * L + y] = phi_new[x][y];
            }
        }

        /* debug for calculation
        if(my_rank == 0){
            printf("relax my_rank 0 phi\n");
            for(int i = 0; i < L * L; i++){
                if(i% L == 0) printf("\n");
                printf("%.1f ", phi[i]);
            }
            printf("\n");
        
            printf("relax my_rank 0 res\n");
            for(int i = 0; i < L * L; i++){
                if(i% L == 0) printf("\n");
                printf("%.1f ", res[i]);
            }
            printf("\n");
        }
        */
    }
    //exit(0);


   //printf("relax line 270\n");
    

    
    /*
    if(my_rank == 1){
        printf("relax my_rank 1\n");
        for(int i = 0; i < L * L; i++){
            if(i% L == 0) printf("\n");
            printf("%.1f ", phi[i]);
        }
        printf("\n");
    }
    */

    for (int i = 0; i < local_size; i++) {
        free(phi_new[i]);
    }
    free(phi_new);
    free(low_buffer);
    free(high_buffer);

   MPI_Barrier(MPI_COMM_WORLD);
   //printf("relax line 297\n");
   return ;
}


void proj_res(double *res_c, double *res_f, double *phi_f, int lev, param_t p)
{
    int L, Lc, f_off, c_off, x, y;
    L = p.size[lev];
    double r[L * L];      // temp residue
    Lc = p.size[lev + 1]; // course level
    // get residue
    for (x = 0; x < L; x++)
        for (y = 0; y < L; y++)
            r[x + y * L] = res_f[x + y * L] - phi_f[x + y * L] + p.scale[lev] * (phi_f[(x + 1) % L + y * L] + phi_f[(x - 1 + L) % L + y * L] + phi_f[x + ((y + 1) % L) * L] + phi_f[x + ((y - 1 + L) % L) * L]);

    // project residue
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

    for (x = 0; x < Lc; x++)
        for (y = 0; y < Lc; y++)
        {
            phi_f[2 * x + 2 * y * L] += phi_c[x + y * Lc];
            phi_f[(2 * x + 1) % L + 2 * y * L] += phi_c[x + y * Lc];
            phi_f[2 * x + ((2 * y + 1)) % L * L] += phi_c[x + y * Lc];
            phi_f[(2 * x + 1) % L + ((2 * y + 1) % L) * L] += phi_c[x + y * Lc];
        }
    // set to zero so phi = error
    for (x = 0; x < Lc; x++)
        for (y = 0; y < Lc; y++)
            phi_c[x + y * Lc] = 0.0;

    return;
}

double GetResRoot(double *phi, double *res, int lev, param_t p)
{ // true residue
    int i, x, y;
    double residue;
    double ResRoot = 0.0;
    int L;
    L = p.size[lev];

    for (x = 0; x < L; x++)
        for (y = 0; y < L; y++)
        {
            residue = res[x + y * L] / p.scale[lev] - phi[x + y * L] / p.scale[lev] + (phi[(x + 1) % L + y * L] + phi[(x - 1 + L) % L + y * L] + phi[x + ((y + 1) % L) * L] + phi[x + ((y - 1 + L) % L) * L]);
            ResRoot += residue * residue; // true residue
        }

    return sqrt(ResRoot);
}
