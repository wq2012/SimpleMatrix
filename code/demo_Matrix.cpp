#include <iostream>
#include "SimpleMatrix.h"

using namespace std;

int main(int argc, const char *argv[])
{
  // create matrix of same values
  smat::Matrix<double> A(3, 4, 1.0);
  printf("A is a 3*4 matrix of the same value 1, A=\n");
  A.print();

  // create matrix of random values
  smat::Matrix<double> B(5, 4, "rand");
  printf("B is a 5*4 matrix of random values in [0,1], B=\n");
  B.print();

  // read matrix from txt file
  smat::Matrix<double> C("X.txt");
  printf("C is a matrix read from \"X.txt\", C=\n");
  C.print();

  // matrix multiplication
  smat::Matrix<double> P = A.multiplyMatrixNew(C);
  printf("P=A*C, P=\n");
  P.print();

  ///////////////////////////////////////////////////
  // multidimensional scaling (MDS) experiment     //
  ///////////////////////////////////////////////////

  // input D is the geodesic distance matrix of a Swiss roll surface, and output X is the flattened surface
  smat::Matrix<double> D("swissD.txt");   // read the distance matrix from txt file
  smat::Matrix<double> X0("swissX0.txt"); // read the initialization matrix from txt file
  int dim = 3;                                                        // number of dimensions
  int iter = 30;                                                      // number of iterations

  printf("\n============================================\n");
  printf("Begin to run MDS(UCF): \n");
  printf("============================================\n");

  smat::Matrix<double> X1 = smat::MDS_UCF(D, NULL, dim, iter); // without initialization
  printf("[PASS] MDS(UCF) done! \n");
  X1.saveTxt("swissX1.txt");

  printf("\n============================================\n");
  printf("Begin to run MDS(SMACOF): \n");
  printf("============================================\n");
  smat::Matrix<double> X2 = smat::MDS_SMACOF(D, NULL, dim, iter); // without initialization
  printf("[PASS] MDS(SMACOF) done! \n");
  X2.saveTxt("swissX2.txt");

  printf("\n============================================\n");
  printf("Begin to run MDS(UCF) with initialization: \n");
  printf("============================================\n");
  smat::Matrix<double> X3 = smat::MDS_UCF(D, &X0, dim, iter); // with initialization
  printf("[PASS] MDS(UCF) done! \n");
  X3.saveTxt("swissX3.txt");

  printf("\n============================================\n");
  printf("Begin to run MDS(SMACOF) with initialization: \n");
  printf("============================================\n");
  smat::Matrix<double> X4 = smat::MDS_SMACOF(D, &X0, dim, iter); // with initialization
  printf("[PASS] MDS(SMACOF) done! \n");
  X4.saveTxt("swissX4.txt");

  return 0;
}
