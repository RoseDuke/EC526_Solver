.SUFFIXES:
.SUFFIXES: .o .cpp

#============================================================
TARGETS =  mg_omp

C_SOURCES = mg_omp.cpp 
C_OBJS    = mg_omp.o 
MY_INCLUDES =

CCX = g++
CXXFLAGS = -g -O3 -fopenmp

#============================================================
all: $(TARGETS)

%.o: %.cpp $(MY_INCLUDES)
	$(CCX) -c $(CXXFLAGS) $< -o $@

mg_omp: mg_omp.o
	$(CCX) $(CXXFLAGS) $^ -o $@

#============================================================

ALL_SOURCES = Makefile $(C_SOURCES) $(MY_INCLUDES)

NOTES = FFT_2021.pdf

clean:
	rm -f $(TARGETS) $(C_OBJS) *.*~ core 

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGETS).tar $(ALL_SOURCES) $(NOTES)
