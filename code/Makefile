CC = g++
CFLAGS = -Wall -std=c++11

.PHONY: all clean test run

all: clean demo_Matrix.exe SimpleMatrixTest.exe

run: demo_Matrix.exe
	./demo_Matrix.exe

test: SimpleMatrixTest.exe
	./SimpleMatrixTest.exe

demo_Matrix.exe: demo_Matrix.cpp SimpleMatrix.h
	$(CC) $(CFLAGS) demo_Matrix.cpp -o demo_Matrix.exe

SimpleMatrixTest.exe: SimpleMatrixTest.cpp SimpleMatrix.h
	$(CC) $(CFLAGS) SimpleMatrixTest.cpp -o SimpleMatrixTest.exe

clean:
	rm -f demo_Matrix.exe SimpleMatrixTest.exe
