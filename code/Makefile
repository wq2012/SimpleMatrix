CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -O2
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2
LDFLAGS = -lm

TARGETS = demo_Matrix.exe demo_MatrixC.exe SimpleMatrixTest.exe SimpleMatrixTestC.exe \
          example_linear_systems.exe example_decompositions.exe example_data_science.exe example_c_usage.exe

.PHONY: all clean test run examples

all: clean $(TARGETS)

run: demo_Matrix.exe demo_MatrixC.exe
	./demo_Matrix.exe
	./demo_MatrixC.exe

test: SimpleMatrixTest.exe SimpleMatrixTestC.exe
	./SimpleMatrixTest.exe
	./SimpleMatrixTestC.exe

examples: example_linear_systems.exe example_decompositions.exe example_data_science.exe example_c_usage.exe
	./example_linear_systems.exe
	./example_decompositions.exe
	./example_data_science.exe
	./example_c_usage.exe

demo_Matrix.exe: demo_Matrix.cpp SimpleMatrix.h
	$(CXX) $(CXXFLAGS) demo_Matrix.cpp -o demo_Matrix.exe

demo_MatrixC.exe: demo_MatrixC.c SimpleMatrix.h
	$(CC) $(CFLAGS) demo_MatrixC.c $(LDFLAGS) -o demo_MatrixC.exe

SimpleMatrixTest.exe: SimpleMatrixTest.cpp SimpleMatrix.h
	$(CXX) $(CXXFLAGS) SimpleMatrixTest.cpp -o SimpleMatrixTest.exe

SimpleMatrixTestC.exe: SimpleMatrixTestC.c SimpleMatrix.h
	$(CC) $(CFLAGS) SimpleMatrixTestC.c $(LDFLAGS) -o SimpleMatrixTestC.exe

example_linear_systems.exe: examples/example_linear_systems.cpp SimpleMatrix.h
	$(CXX) $(CXXFLAGS) examples/example_linear_systems.cpp -o example_linear_systems.exe

example_decompositions.exe: examples/example_decompositions.cpp SimpleMatrix.h
	$(CXX) $(CXXFLAGS) examples/example_decompositions.cpp -o example_decompositions.exe

example_data_science.exe: examples/example_data_science.cpp SimpleMatrix.h
	$(CXX) $(CXXFLAGS) examples/example_data_science.cpp -o example_data_science.exe

example_c_usage.exe: examples/example_c_usage.c SimpleMatrix.h
	$(CC) $(CFLAGS) examples/example_c_usage.c $(LDFLAGS) -o example_c_usage.exe

clean:
	rm -f $(TARGETS) swissX1.txt swissX2.txt swissX3.txt swissX4.txt
