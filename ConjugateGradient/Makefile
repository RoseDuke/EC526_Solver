CC = mpicxx
CFLAGS = -g -Wall
TARGET = mpi_CG_solver
SRC = mpi_CG_solver.cpp

all:$(TARGET)

$(TARGET):$(SRC)
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -f $(TARGET)