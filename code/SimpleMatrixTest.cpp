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

void test_linear_algebra()
{
  // Test Determinant
  smat::Matrix<double> *A = new smat::Matrix<double>(3, 3);
  // 1 2 3
  // 4 5 6  (singular det=0) -> make it non-singular: 6->7
  // 7 8 7   
  // det = 1(35-48) - 2(28-42) + 3(32-35) = 1(-13) - 2(-14) + 3(-3) = -13 + 28 - 9 = 6
  A->set(0,0,1); A->set(0,1,2); A->set(0,2,3);
  A->set(1,0,4); A->set(1,1,5); A->set(1,2,6); // 4 5 6
  A->set(2,0,7); A->set(2,1,8); A->set(2,2,7); // 7 8 7
  
  assertFloatEqual(A->determinant(), 6.0);
  
  // Test Inverse
  smat::Matrix<double> *Inv = A->inverse();
  smat::Matrix<double> I = (*A) * (*Inv);
  // Check if I is identity
  for(int i=0; i<3; i++)
      for(int j=0; j<3; j++)
          assertFloatEqual(I.get(i, j), (i==j)?1.0:0.0);
  delete Inv;

  // Test LU
  smat::Matrix<double> *L, *U, *P;
  A->lu(L, U, P);
  
  smat::Matrix<double> PA = (*P) * (*A);
  smat::Matrix<double> LU = (*L) * (*U);
  
  for(int i=0; i<3; i++)
      for(int j=0; j<3; j++)
          assertFloatEqual(PA.get(i, j), LU.get(i, j));
          
  delete L; delete U; delete P; delete A;
}

void test_qr()
{
  smat::Matrix<double> *A = new smat::Matrix<double>(3, 3);
  // 12 -51 4
  // 6 167 -68
  // -4 24 -41
  A->set(0,0,12); A->set(0,1,-51); A->set(0,2,4);
  A->set(1,0,6); A->set(1,1,167); A->set(1,2,-68);
  A->set(2,0,-4); A->set(2,1,24); A->set(2,2,-41);
  
  smat::Matrix<double> *Q, *R;
  A->qr(Q, R);
  
  // Verify A = QR
  smat::Matrix<double> QR = (*Q) * (*R);
  for(int i=0; i<3; i++)
     for(int j=0; j<3; j++)
         assertFloatEqual(A->get(i, j), QR.get(i, j));
         
  // Verify Q is orthogonal (Q^T Q = I)
  smat::Matrix<double> *Qt = Q->transpose();
  smat::Matrix<double> QtQ = (*Qt) * (*Q);
  for(int i=0; i<3; i++)
     for(int j=0; j<3; j++)
         assertFloatEqual(QtQ.get(i, j), (i==j)?1.0:0.0);
         
  // Verify R is upper triangular
  assertFloatEqual(R->get(1, 0), 0.0);
  assertFloatEqual(R->get(2, 0), 0.0);
  assertFloatEqual(R->get(2, 1), 0.0);
  
  delete Q; delete R; delete Qt; delete A;
}

void test_eigen()
{
  smat::Matrix<double> *A = new smat::Matrix<double>(3, 3);
  // Symmetric Matrix
  // 2 -1 0
  // -1 2 -1
  // 0 -1 2
  A->set(0,0,2); A->set(0,1,-1); A->set(0,2,0);
  A->set(1,0,-1); A->set(1,1,2); A->set(1,2,-1);
  A->set(2,0,0); A->set(2,1,-1); A->set(2,2,2);
  
  smat::Matrix<double> *V, *D;
  A->eigen(V, D);
  
  // Verify A V = V D
  smat::Matrix<double> AV = (*A) * (*V);
  smat::Matrix<double> VD = (*V) * (*D);
  
  for(int i=0; i<3; i++)
     for(int j=0; j<3; j++)
         assertFloatEqual(AV.get(i, j), VD.get(i, j));
         
  // Verify VT V = I (Orthogonal)
  smat::Matrix<double> *Vt = V->transpose();
  smat::Matrix<double> VtV = (*Vt) * (*V);
  for(int i=0; i<3; i++)
     for(int j=0; j<3; j++)
         assertFloatEqual(VtV.get(i, j), (i==j)?1.0:0.0);
         
  // Check eigenvalues (known: 2-sqrt(2), 2, 2+sqrt(2) => 0.5857, 2, 3.4142)
  // But order is arbitrary. Just print or checking diagonal is enough.
  
  delete V; delete D; delete Vt; delete A;
}

void test_svd()
{
  smat::Matrix<double> *A = new smat::Matrix<double>(3, 2);
  // 1 2
  // 3 4
  // 5 6
  A->set(0,0,1); A->set(0,1,2);
  A->set(1,0,3); A->set(1,1,4);
  A->set(2,0,5); A->set(2,1,6);
  
  smat::Matrix<double> *U, *S, *Vt;
  A->svd(U, S, Vt);
  
  // Verify A = U S Vt
  smat::Matrix<double> US = (*U) * (*S);
  smat::Matrix<double> USVt = US * (*Vt);
  
  for(int i=0; i<3; i++)
     for(int j=0; j<2; j++)
         assertFloatEqual(A->get(i, j), USVt.get(i, j));
         
  // Verify U is orthogonal-ish (columns are orthonormal)
  // U is m x n (Compact SVD) -> U^T U = I
  smat::Matrix<double> *Ut = U->transpose();
  smat::Matrix<double> UtU = (*Ut) * (*U);
  for(int i=0; i<2; i++)
     for(int j=0; j<2; j++)
         assertFloatEqual(UtU.get(i, j), (i==j)?1.0:0.0);
         
  // Verify V is orthogonal
  // Vt is V^T. Vt * Vt^T = I
  smat::Matrix<double> *V = Vt->transpose();
  smat::Matrix<double> VVt = (*V) * (*Vt); // Or Vt * V
  for(int i=0; i<2; i++)
     for(int j=0; j<2; j++)
         assertFloatEqual(VVt.get(i, j), (i==j)?1.0:0.0);
         
  delete U; delete S; delete Vt; delete Ut; delete V; delete A;
}

void test_rank()
{
  smat::Matrix<double> *A = new smat::Matrix<double>(3, 3);
  // Full rank
  // 1 0 0 
  // 0 1 0
  // 0 0 1
  A->set(0,0,1); A->set(1,1,1); A->set(2,2,1);
  if (A->rank() != 3) { printf("Rank check failed (expected 3, got %d)\n", A->rank()); exit(1); }
  
  // Rank deficient
  // 1 1 1
  // 1 1 1
  // 1 1 1
  // Rank should be 1
  for(int i=0; i<3; i++) for(int j=0; j<3; j++) A->set(i, j, 1.0);
  if (A->rank() != 1) { printf("Rank check failed (expected 1, got %d)\n", A->rank()); exit(1); }
  
  delete A;
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
  test_linear_algebra();
  test_qr();
  test_eigen();
  test_svd();
  test_rank();
  std::cout << "All tests passed!" << std::endl;
  return 0;
}
