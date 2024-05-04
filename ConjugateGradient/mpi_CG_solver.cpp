#include <iostream>
#include <math.h>
#include <stack>
#include <chrono>
#include <mpi.h>
#include <strings.h>

using namespace std;

#define Lx  512 // 16 
#define Ly  512 // 16
#define N   Lx * Ly

#define max(a,b) (a>b?a:b)
#define min(a,b) (a>b?b:a)

inline int mod(int x, int n)
{
  return  ((x % n) + n )%n;
}

inline int nn(int site,int mu)
{
  int neighbor;
  int x,y;
  
  x = site%Lx; y = site/Lx;
  
  switch(mu){
  case 0:  neighbor =  (x+1)%Lx + Lx*y;break; // x + 1
  case 1:  neighbor =  (x-1 + Lx)%Lx + Lx*y; break;  // x -1  
  case 2:  neighbor =  x + Lx*((y+1)%Ly);  break; // y + 1
  case 3:  neighbor =  x + Lx*((y-1+Ly)%Ly); break;  // y - 1 
  default: neighbor = -1;
  }

  //printf("my neighbor is on %d\n", neighbor%Lx);

    return neighbor; 
}

/******  Without preconditing *********/

void exchange_ghost_zone(double* local_x, int local_rows)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Request reqs[4];

    MPI_Isend(&local_x[Lx], Lx, MPI_DOUBLE, (rank-1+size)%size, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&local_x[0], Lx, MPI_DOUBLE, (rank-1+size)%size, 1, MPI_COMM_WORLD, &reqs[1]);
    // Send bottom row to the next rank and receive the top ghost zone from the next rank
    MPI_Isend(&local_x[Lx * local_rows], Lx, MPI_DOUBLE, (rank+1)%size, 1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(&local_x[Lx * (local_rows + 1)], Lx, MPI_DOUBLE, (rank+1)%size, 0, MPI_COMM_WORLD, &reqs[3]);

    MPI_Waitall(rank > 0 && rank < size - 1 ? 4 : 2, reqs, MPI_STATUSES_IGNORE);
}

int Avec(double * vec_out, double * vec_in, int local_rows)
{
  double mass = 0.01;
  for(int i = Lx; i < Lx*(local_rows+1); i++)
    {
      //printf("I'm %d\n", i);
      vec_out[i] = (1.0 + mass*mass)* vec_in[i];
        for(int mu = 0; mu < 4; mu++)
	  {
       	   vec_out[i] = vec_out[i] - 0.25 * vec_in[nn(i, mu)];				   
	}
      //printf("\n");
    }   
  return 0;
}

double dot(double *  v1, double  * v2, int local_rows)
{
  double scalar = 0.0;
  
  for(int site = Lx; site < Lx*(local_rows+1); site++)
    scalar += v1[site]*v2[site];
  
  //MPI_Reduce(&scalart, &global_scalar, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

  return scalar;
}

int   CGiter(double * x, double * b,  int local_rows, double rs_stop, int rank)
{
    double local_r[Lx*(local_rows+2)], local_p[Lx*(local_rows+2)];
    double local_Ax[Lx*(local_rows+2)], local_Ap[Lx*(local_rows+2)];
    double alpha,beta;
    double local_rsold, global_rsold, local_rsnew, global_rsnew;
    double local_pdot, global_pdot;
    int iter = -1;
    int MaxIter = 100000;
    int continue_iteration = 1;

    // Intialize
    exchange_ghost_zone(x, local_rows);
    Avec(local_Ax, x, local_rows);
    for(int i = Lx; i < Lx*(local_rows+1); i++)
    {
        local_r[i] = b[i] - local_Ax[i];
        local_p[i] = local_r[i]; 
    }
    
    double global_dot = 0;
    double local_dot = dot(local_r, local_r, local_rows);
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
      if (global_dot < rs_stop) 
      {
        continue_iteration = 0;
      }
    }
    MPI_Bcast(&continue_iteration, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (continue_iteration == 0) {
      return iter;
    }
    
  
    for(iter = 0 ; iter <MaxIter; iter++)
    {
        local_rsold = dot(local_r,local_r, local_rows);
        MPI_Allreduce(&local_rsold, &global_rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        exchange_ghost_zone(local_p, local_rows);
        Avec(local_Ap, local_p, local_rows);

        local_pdot = dot(local_p, local_Ap, local_rows);
        MPI_Allreduce(&local_pdot, &global_pdot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        alpha = global_rsold / global_pdot;
      
        for(int i = Lx ; i < Lx*(local_rows+1) ; i++)
	      {
	        x[i] = x[i] + alpha * local_p[i];
	        local_r[i] = local_r[i] - alpha * local_Ap[i];
	      }
      
        local_rsnew = dot(local_r,local_r, local_rows);
        MPI_Allreduce(&local_rsnew, &global_rsnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        beta = global_rsnew/global_rsold;

        if(rank == 0)
        {
          if (global_rsnew < rs_stop) 
          {
            continue_iteration = 0;
          }
        }
        MPI_Bcast(&continue_iteration, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (continue_iteration == 0) {
          break;
        }

        for(int i = Lx ; i < Lx*(local_rows+1) ; i++)
	        local_p[i] = local_r[i] + beta* local_p[i];
        //printf("I'm %d and my label is %d\n", rank, continue_iteration);
    }
    //printf("I'm %d and my label is %d\n", rank, continue_iteration);
    return iter;     
}
 
void printArray(double * phi)
{
	cout<<"\n--------------------------------------------";
	for(int y = 0; y<Ly; y++)
	  {
	     cout << endl;
	    for(int x= 0 ; x<Lx; x++)
   	      printf(" %10.5f ", phi[x + y* Lx]);    
	  }
	cout<<"\n-------------------------------------------- \n";
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  int local_rows = Ly / size;
  //Parameters for collective phi
  double output_phi[Lx*Ly];
  double send_buffer[Lx*local_rows];
  int sendcount = Lx*local_rows;
  int recvcount[size];
  int place[size];
  for(int i=0;i<size;i++)
  {
    recvcount[i] = Lx*local_rows;
    place[i] = i*Lx*local_rows;
  }

  auto start = std::chrono::high_resolution_clock::now();

  FILE* outfile_mpi;
  if (rank == 0) {
        outfile_mpi = fopen("phi_mpi_periodic_boundary.txt", "w");
        if (!outfile_mpi) {
            std::cerr << "Failed to open file for writing." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);  // Handle error appropriately
        }
    }

  printf(" Lattice %d by %d  with %d sites \n", Lx, Ly,N);

  double  local_phi[Lx*(local_rows+2)], local_b[Lx*(local_rows+2)]; //Allocation

  srand(137);
 
  for(int i = 0; i < Lx*(local_rows+2) ; i++) {
    local_phi[i] = 0.0;
    local_b[i] = 0.0;
  }
  
  if(rank == size / 2){
    local_b[Lx + Lx/4] = 1.0; //After the first ghost zone, the source point is at the middle of first row
    local_b[Lx + 3*Lx/4] = -1.0;
  }
  /*
  if(rank == 0) printf("I'm %d before\n", rank);
  if(rank == 1) printf("I'm %d before\n", rank);
  if(rank == 2) printf("I'm %d before\n", rank);
  if(rank == 3) printf("I'm %d before\n", rank);
  */
  double  rms_stop =  1.0e-06;
  int numOfiter = 0 ;
  numOfiter = CGiter(local_phi,  local_b, local_rows, rms_stop, rank);

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

  //printArray(phi);
  cout << " Number of interation of CG = " <<  numOfiter << " at rms_stop = " <<  rms_stop   <<  endl;
  /*
  for(int i=0;i<Lx;i++){
    for(int j=0;j<Ly;j++){
        fprintf(outfile_serial, "%f ", phi[i+j*Lx]);
    }
    fprintf(outfile_serial, "\n");
  }
  fclose(outfile_serial);
  */

  // Gather the data at the root process
  /*
  if(rank == 0) printf("I'm %d after\n", rank);
  if(rank == 1) printf("I'm %d after\n", rank);
  if(rank == 2) printf("I'm %d after\n", rank);
  if(rank == 3) printf("I'm %d after\n", rank);
  */
  memcpy(send_buffer, local_phi+Lx, Lx*local_rows*sizeof(double));
  MPI_Gatherv(send_buffer, sendcount, MPI_DOUBLE,
                output_phi, recvcount, place, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Data gathered at root:\n";
      for (int i = 0; i < Lx * local_rows * size; i++) {
        fprintf(outfile_mpi, "%f  ", output_phi[i]);
        if ((i + 1) % Lx == 0) fprintf(outfile_mpi, "\n");
      }
    }
  
  MPI_Finalize();
  return 0;
}
