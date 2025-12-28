.PHONY: all clean

all: demo_Matrix.exe SimpleMatrixTest.exe

demo_Matrix.exe: demo_Matrix.cpp SimpleMatrix.h
	g++ -Wall demo_Matrix.cpp -o demo_Matrix.exe

SimpleMatrixTest.exe: SimpleMatrixTest.cpp SimpleMatrix.h
	g++ -Wall SimpleMatrixTest.cpp -o SimpleMatrixTest.exe

clean:
	rm -f demo_Matrix.exe SimpleMatrixTest.exe
