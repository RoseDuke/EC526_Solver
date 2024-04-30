CXX=mpicxx

CXXFLAGS=-std=c++11

LIBS=-lstdc++

TARGET=mg

SOURCES=mg_mpi.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $^ $(LIBS)

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
