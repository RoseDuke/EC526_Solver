CC=pgc++

CFLAGS=-g -Wall -std=c++11 -Minfo=accel -fast -acc -ta=tesla:ccall

LDFLAGS=-acc -ta=tesla:ccall -Mcuda

OBJS1=mg_acc.o

TARGET1=mg_acc

all: $(TARGET1)

$(TARGET1): $(OBJS1)
	$(CC) $(LDFLAGS) -o $(TARGET1) $(OBJS1)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS1) $(TARGET1)
