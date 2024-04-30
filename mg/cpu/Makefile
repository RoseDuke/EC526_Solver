.SUFFIXES:
.SUFFIXES: .o .cpp

#============================================================
TARGETS =  mg

C_SOURCES = mg.cpp 
C_OBJS    = mg.o 
MY_INCLUDES =

CCX = g++
CXXFLAGS = -g -O2

#============================================================
all: $(TARGETS)

%.o: %.cpp $(MY_INCLUDES)
	$(CCX) -c $(CXXFLAGS) $< -o $@

mg: mg.o
	$(CCX) $(CXXFLAGS) $^ -o $@

#============================================================

ALL_SOURCES = Makefile $(C_SOURCES) $(MY_INCLUDES)

NOTES = FFT_2021.pdf

clean:
	rm -f $(TARGETS) $(C_OBJS) *.*~ core 

tar: $(ALL_SOURCES) $(NOTES)
	tar cvf $(TARGETS).tar $(ALL_SOURCES) $(NOTES)
