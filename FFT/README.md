# FFT method to solve Heat Transfer Problem

How to compile on SCC

---

#### Step1: apply for a Nvidia-V100

~~~bash
qrsh -l gpus=1 -l gpu_c=7.0 -pe omp 4
~~~

#### Step2: set up environment

~~~bash
module load pgi
module load gcc
~~~

#### Step3: compile and run

~~~bash
make -f Makefile.txt
./fft
~~~

And all the data will go to ./output.dat .