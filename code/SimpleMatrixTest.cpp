#include <cassert>
#include <cmath>
#include <iostream>
#include "SimpleMatrix.h"

const float eps = 0.001;

void assertIntEq(int x, int y)
{
  if (x != y)
  {
    std::cerr << "assertIntEq failed for " << x << " and " << y << std::endl;
    assert(false);
  }
}

void assertFloatEqual(float x, float y, float threshold = eps)
{

  if (fabs(x - y) > threshold)
  {
    std::cerr << "assertFloatEqual failed for "
              << x << " and " << y << " with threshold " << threshold
              << std::endl;
    assert(false);
  }
}

void test_basics()
{
  smat::Matrix<double> *A = new smat::Matrix<double>(3, 4, 1);
  assertIntEq(A->get(2, 3), 1);
  A->set(2, 3, 5);
  assertIntEq(A->get(2, 3), 5);
  assertIntEq(A->get(1, 1), 1);
  assertIntEq(A->rows(), 3);
  assertIntEq(A->columns(), 4);
  delete A;
}

void test_statistics()
{
  smat::Matrix<double> *A = new smat::Matrix<double>(3, 4, 1);
  A->set(2, 3, 5);
  A->set(1, 1, -2); // A now has elements: most 1, one 5, one -2.

  // sum was 12 (3*4) -> after set(5) (+4) -> 16 -> after set(-2) (-3) -> 13
  assertIntEq(A->sum(), 13);
  assertIntEq(A->trace(), 0); // Diagonals: (0,0)=1, (1,1)=-2, (2,2)=1 -> 1-2+1 = 0

  int r, c;
  int max_el = A->maxEl(r, c);
  assertIntEq(max_el, 5);
  assertIntEq(r, 2);
  assertIntEq(c, 3);

  int min_el = A->minEl(r, c);
  assertIntEq(min_el, -2);
  assertIntEq(r, 1);
  assertIntEq(c, 1);

  // Mean: 13 / 12 = 1.08333
  assertFloatEqual(A->mean(), 1.08333);

  // Fnorm: sqrt(10*1^2 + 5^2 + (-2)^2) = sqrt(10 + 25 + 4) = sqrt(39) = 6.245
  assertFloatEqual(A->fnorm(), 6.245);

  delete A;
}

void test_scalar_operations()
{
  smat::Matrix<double> *A = new smat::Matrix<double>(2, 2, 4.0);
  
  A->addNumberSelf(1.0);
  assertFloatEqual(A->get(0, 0), 5.0);
  
  A->subtractNumberSelf(1.0);
  assertFloatEqual(A->get(0, 0), 4.0);
  
  A->divideNumberSelf(2.0);
  assertFloatEqual(A->get(0, 0), 2.0);
  
  A->multiplyNumberSelf(3.0);
  assertFloatEqual(A->get(0, 0), 6.0);
  
  delete A;
}

void test_math_functions()
{
  smat::Matrix<double> *D = new smat::Matrix<double>(1, 1, 100.0);
  
  smat::Matrix<double> *SqrtD = D->sqrt();
  assertFloatEqual(SqrtD->get(0, 0), 10.0);
  delete SqrtD;

  smat::Matrix<double> *E = new smat::Matrix<double>(1, 1, 1.0);
  smat::Matrix<double> *LogE = E->log(); // log(1) = 0
  assertFloatEqual(LogE->get(0, 0), 0.0);
  smat::Matrix<double> *ExpLogE = LogE->exp(); // exp(0) = 1
  assertFloatEqual(ExpLogE->get(0, 0), 1.0);
  
  delete E;
  delete LogE;
  delete ExpLogE;
  delete D;
}

void test_matrix_arithmetic()
{
  smat::Matrix<double> *A = new smat::Matrix<double>(2, 2, 4.0);
  smat::Matrix<double> *B = new smat::Matrix<double>(2, 2, 2.0);

  // Subtraction
  smat::Matrix<double> *C = A->subtractMatrixNew(B);
  assertFloatEqual(C->get(0, 0), 2.0);
  delete C;
  A->subtractMatrixSelf(B);
  assertFloatEqual(A->get(0, 0), 2.0); // A becomes 2

  // Division
  A->set(0, 0, 10.0);
  B->set(0, 0, 2.0);
  C = A->dotDivideMatrixNew(B);
  assertFloatEqual(C->get(0, 0), 5.0);
  delete C;
  A->dotDivideMatrixSelf(B);
  assertFloatEqual(A->get(0, 0), 5.0);

  delete A;
  delete B;
}

void test_concatenation()
{
  smat::Matrix<double> *E = new smat::Matrix<double>(2, 2, 1.0);
  smat::Matrix<double> *F = new smat::Matrix<double>(2, 2, 2.0);
  
  smat::Matrix<double> *H = E->concatenateRight(F);
  assertIntEq(H->rows(), 2);
  assertIntEq(H->columns(), 4);
  assertFloatEqual(H->get(0, 3), 2.0);
  assertFloatEqual(H->get(0, 0), 1.0);
  delete H;

  smat::Matrix<double> *V = E->concatenateBottom(F);
  assertIntEq(V->rows(), 4);
  assertIntEq(V->columns(), 2);
  assertFloatEqual(V->get(3, 0), 2.0);
  assertFloatEqual(V->get(0, 0), 1.0);
  delete V;

  delete E;
  delete F;
}

void test_operators()
{
  smat::Matrix<double> M1(2, 2, 3.0);
  smat::Matrix<double> M2(2, 2, 2.0);
  
  smat::Matrix<double> M3 = M1 + M2;
  assertFloatEqual(M3.get(0, 0), 5.0);
  
  smat::Matrix<double> M4 = M1 * M2; // [3 3; 3 3] * [2 2; 2 2] = [12 12; 12 12]
  assertFloatEqual(M4.get(0, 0), 12.0);
  
  smat::Matrix<double> M5 = M1 - M2;
  assertFloatEqual(M5.get(0, 0), 1.0);
}

void test_mds()
{
  // Create a distance matrix for 4 points in a unit square:
  // (0,0), (0,1), (1,0), (1,1)
  // Distance matrix:
  // 0  1  1  sqrt(2)
  // 1  0  sqrt(2) 1
  // 1  sqrt(2) 0  1
  // sqrt(2) 1  1  0
  
  int n = 4;
  smat::Matrix<double> *D = new smat::Matrix<double>(n, n, 0.0);
  double s2 = std::sqrt(2.0);
  
  D->set(0, 1, 1.0); D->set(0, 2, 1.0); D->set(0, 3, s2);
  D->set(1, 0, 1.0); D->set(1, 2, s2);  D->set(1, 3, 1.0);
  D->set(2, 0, 1.0); D->set(2, 1, s2);  D->set(2, 3, 1.0);
  D->set(3, 0, s2);  D->set(3, 1, 1.0); D->set(3, 2, 1.0);

  // Run MDS
  // Use a fixed initialization for determinism.
  int dim = 2;
  int iter = 50; 
  
  smat::Matrix<double> *X0 = new smat::Matrix<double>(n, dim);
  X0->set(0, 0, 0.1); X0->set(0, 1, 0.1);
  X0->set(1, 0, 0.1); X0->set(1, 1, 0.9);
  X0->set(2, 0, 0.9); X0->set(2, 1, 0.1);
  X0->set(3, 0, 0.9); X0->set(3, 1, 0.9);

  smat::Matrix<double> *X = smat::MDS_SMACOF(D, X0, dim, iter);
  
  delete X0;
  
  // Calculate distances of X
  smat::Matrix<double> *D_rec = new smat::Matrix<double>(n, n);
  smat::EuclideanDistanceMatrix(X, D_rec);
  
  // Verify (tolerant checks)
  // MDS reconstruction error (stress) might not be exactly 0 due to local minima,
  // but for a square it should be very close.
  for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {
      assertFloatEqual(D->get(i, j), D_rec->get(i, j), 0.05); // 0.05 tolerance
    }
  }

  delete D;
  delete X;
  delete D_rec;
}

int main(int argc, const char *argv[])
{
  test_basics();
  test_statistics();
  test_scalar_operations();
  test_math_functions();
  test_matrix_arithmetic();
  test_concatenation();
  test_operators();
  test_mds();
  std::cout << "All tests passed!" << std::endl;
  return 0;
}
