#include <iostream>
#include <math.h>
#include <stack>
#include <chrono>
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

int  CGiter(double * x, double * b, int NoIter, double rs_stop);
int Avec(double * vec_out, double * vec_in);
double dot(double * v1,double * v2);
void printArray(double * phi);

int main()
{
  auto start = std::chrono::high_resolution_clock::now();
  FILE* outfile_serial;
  outfile_serial = fopen("phi_serial.txt", "w");
  printf(" Lattice %d by %d  with %d sites \n", Lx, Ly,N);
  double  phi[N], b[N];
  srand(137);
 
  for(int i = 0; i < N ; i++) {
    phi[i] = 0.0;
    b[i] = 0.0;
  }
  
  b[Lx/2 + Lx* (Ly/2)] = 1.0;
  
  double  rms_stop =  1.0e-06;
  int numOfiter = 0 ;
  numOfiter = CGiter(phi,  b, 1000, rms_stop);

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

  //printArray(phi);
  cout << " Number of interation of CG = " <<  numOfiter << " at rms_stop = " <<  rms_stop   <<  endl;

  for(int i=0;i<Lx;i++){
    for(int j=0;j<Ly;j++){
        fprintf(outfile_serial, "%f ", phi[i+j*Lx]);
    }
    fprintf(outfile_serial, "\n");
  }
  fclose(outfile_serial);
  return 0;
}

int Avec(double * vec_out, double * vec_in)
{
  double mass = 1;
  for(int i = 0; i < N; i++)
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

double dot(double *  v1, double  * v2)
{
  double scalar = 0.0;
  
  for(int site = 0; site < N; site++)
    scalar += v1[site]*v2[site];

  return scalar;
}


/******  Without preconditing *********/

int   CGiter(double * x, double * b,  int skip, double rs_stop)
{
  double r[N], p[N];
  double Ax[N], Ap[N];
  double alpha,beta;
  double rsold, rsnew;
  int iter = -1;
  int MaxIter = 100000;

  // Intialize
  Avec(Ax, x);
  for(int i = 0; i < N; i++)
    {
      r[i] = b[i] - Ax[i];
      p[i] =r[i]; 
    }
  
  if (dot(r,r) < rs_stop) return iter;  // no need to iterate!
  
  for(iter = 0 ; iter <MaxIter; iter++)
    {
      rsold = dot(r,r);
      Avec(Ap, p);
      alpha = rsold / dot(p,Ap);
      
      for(int i = 0 ; i < N ; i++)
	{
	  x[i] = x[i] + alpha * p[i];
	  r[i] = r[i] - alpha * Ap[i];
	}
      
      rsnew = dot(r,r);
 
      if (rsnew < rs_stop) break;
      beta = rsnew/rsold;
      
      for(int i = 0 ; i < N ; i++)
	p[i] = r[i] + beta* p[i];
     
    }    
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