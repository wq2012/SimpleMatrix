#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>
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

void assertTrue(bool condition, const char* msg = "Assertion failed")
{
  if (!condition)
  {
    std::cerr << msg << std::endl;
    assert(false);
  }
}

template <class T>
bool isLowerTriangular(const smat::Matrix<T>& M, bool unitDiagonal = false)
{
  for (int i = 0; i < M.rows(); i++)
  {
    for (int j = i + 1; j < M.columns(); j++)
    {
      if (std::abs((double)M.get(i, j)) > eps) return false;
    }
    if (unitDiagonal && std::abs((double)M.get(i, i) - 1.0) > eps) return false;
  }
  return true;
}

template <class T>
bool isUpperTriangular(const smat::Matrix<T>& M)
{
  for (int i = 0; i < M.rows(); i++)
  {
    for (int j = 0; j < i; j++)
    {
      if (std::abs((double)M.get(i, j)) > eps) return false;
    }
  }
  return true;
}

template <class T>
bool isDiagonal(const smat::Matrix<T>& M)
{
  for (int i = 0; i < M.rows(); i++)
  {
    for (int j = 0; j < M.columns(); j++)
    {
      if (i != j && std::abs((double)M.get(i, j)) > eps) return false;
    }
  }
  return true;
}

template <class T>
bool isIdentity(const smat::Matrix<T>& M)
{
  if (M.rows() != M.columns()) return false;
  for (int i = 0; i < M.rows(); i++)
  {
    for (int j = 0; j < M.columns(); j++)
    {
      double expected = (i == j) ? 1.0 : 0.0;
      if (std::abs((double)M.get(i, j) - expected) > eps) return false;
    }
  }
  return true;
}

template <class T>
bool isOrthogonal(const smat::Matrix<T>& M)
{
  smat::Matrix<T> Mt = M.transpose();
  smat::Matrix<T> MtM = Mt * M;
  return isIdentity(MtM);
}


void printTestHeader(const char* name)
{
  printf("\n============================================================\n");
  printf("[TEST] %s\n", name);
  printf("============================================================\n");
}

void printTestPass(const char* name)
{
  printf("[PASS] %s passed.\n", name);
  printf("------------------------------------------------------------\n");
}

void test_basics()
{
  printTestHeader("Basics");
  smat::Matrix<double> A(3, 4, 1.0);
  assertIntEq(A.get(2, 3), 1);
  A.set(2, 3, 5);
  assertIntEq(A.get(2, 3), 5);
  assertIntEq(A.get(1, 1), 1);
  assertIntEq(A.rows(), 3);
  assertIntEq(A.columns(), 4);
  printTestPass("Basics");
}

void test_statistics()
{
  printTestHeader("Statistics");
  smat::Matrix<double> A(3, 4, 1.0);
  A.set(2, 3, 5);
  A.set(1, 1, -2); // A now has elements: most 1, one 5, one -2.

  // sum was 12 (3*4) -> after set(5) (+4) -> 16 -> after set(-2) (-3) -> 13
  assertIntEq(A.sum(), 13);
  assertIntEq(A.trace(), 0); // Diagonals: (0,0)=1, (1,1)=-2, (2,2)=1 -> 1-2+1 = 0

  int r, c;
  int max_el = A.maxEl(r, c);
  assertIntEq(max_el, 5);
  assertIntEq(r, 2);
  assertIntEq(c, 3);

  int min_el = A.minEl(r, c);
  assertIntEq(min_el, -2);
  assertIntEq(r, 1);
  assertIntEq(c, 1);

  // Mean: 13 / 12 = 1.08333
  assertFloatEqual(A.mean(), 1.08333);

  // Fnorm: sqrt(10*1^2 + 5^2 + (-2)^2) = sqrt(10 + 25 + 4) = sqrt(39) = 6.245
  assertFloatEqual(A.fnorm(), 6.245);

  printTestPass("Statistics");
}

void test_scalar_operations()
{
  printTestHeader("Scalar Operations");
  smat::Matrix<double> A(2, 2, 4.0);
  
  A.addNumberSelf(1.0);
  assertFloatEqual(A.get(0, 0), 5.0);
  
  A.subtractNumberSelf(1.0);
  assertFloatEqual(A.get(0, 0), 4.0);
  
  A.divideNumberSelf(2.0);
  assertFloatEqual(A.get(0, 0), 2.0);
  
  A.multiplyNumberSelf(3.0);
  assertFloatEqual(A.get(0, 0), 6.0);
  
  printTestPass("Scalar Operations");
}

void test_math_functions()
{
  printTestHeader("Math Functions");
  smat::Matrix<double> D(1, 1, 100.0);
  
  smat::Matrix<double> SqrtD = D.sqrt();
  assertFloatEqual(SqrtD.get(0, 0), 10.0);

  smat::Matrix<double> E(1, 1, 1.0);
  smat::Matrix<double> LogE = E.log(); // log(1) = 0
  assertFloatEqual(LogE.get(0, 0), 0.0);
  smat::Matrix<double> ExpLogE = LogE.exp(); // exp(0) = 1
  assertFloatEqual(ExpLogE.get(0, 0), 1.0);
  
  printTestPass("Math Functions");
}

void test_matrix_arithmetic()
{
  printTestHeader("Matrix Arithmetic");
  smat::Matrix<double> A(2, 2, 4.0);
  smat::Matrix<double> B(2, 2, 2.0);

  // Subtraction
  smat::Matrix<double> C = A.subtractMatrixNew(B);
  assertFloatEqual(C.get(0, 0), 2.0);

  A.subtractMatrixSelf(B);
  assertFloatEqual(A.get(0, 0), 2.0); // A becomes 2

  // Division
  A.set(0, 0, 10.0);
  B.set(0, 0, 2.0);
  C = A.dotDivideMatrixNew(B);
  assertFloatEqual(C.get(0, 0), 5.0);

  A.dotDivideMatrixSelf(B);
  assertFloatEqual(A.get(0, 0), 5.0);

  printTestPass("Matrix Arithmetic");
}

void test_concatenation()
{
  printTestHeader("Concatenation");
  smat::Matrix<double> E(2, 2, 1.0);
  smat::Matrix<double> F(2, 2, 2.0);
  
  smat::Matrix<double> H = E.concatenateRight(F);
  assertIntEq(H.rows(), 2);
  assertIntEq(H.columns(), 4);
  assertFloatEqual(H.get(0, 3), 2.0);
  assertFloatEqual(H.get(0, 0), 1.0);

  smat::Matrix<double> V = E.concatenateBottom(F);
  assertIntEq(V.rows(), 4);
  assertIntEq(V.columns(), 2);
  assertFloatEqual(V.get(3, 0), 2.0);
  assertFloatEqual(V.get(0, 0), 1.0);

  printTestPass("Concatenation");
}

void test_operators()
{
  printTestHeader("Operators");
  smat::Matrix<double> M1(2, 2, 3.0);
  smat::Matrix<double> M2(2, 2, 2.0);
  
  smat::Matrix<double> M3 = M1 + M2;
  assertFloatEqual(M3.get(0, 0), 5.0);
  
  smat::Matrix<double> M4 = M1 * M2; // [3 3; 3 3] * [2 2; 2 2] = [12 12; 12 12]
  assertFloatEqual(M4.get(0, 0), 12.0);
  
  smat::Matrix<double> M5 = M1 - M2;
  assertFloatEqual(M5.get(0, 0), 1.0);
  printTestPass("Operators");
}

void test_mds()
{
  printTestHeader("MDS");
  // Create a distance matrix for 4 points in a unit square:
  // (0,0), (0,1), (1,0), (1,1)
  // Distance matrix:
  // 0  1  1  sqrt(2)
  // 1  0  sqrt(2) 1
  // 1  sqrt(2) 0  1
  // sqrt(2) 1  1  0
  
  int n = 4;
  smat::Matrix<double> D(n, n, 0.0);
  double s2 = std::sqrt(2.0);
  
  D.set(0, 1, 1.0); D.set(0, 2, 1.0); D.set(0, 3, s2);
  D.set(1, 0, 1.0); D.set(1, 2, s2);  D.set(1, 3, 1.0);
  D.set(2, 0, 1.0); D.set(2, 1, s2);  D.set(2, 3, 1.0);
  D.set(3, 0, s2);  D.set(3, 1, 1.0); D.set(3, 2, 1.0);

  // Run MDS
  // Use a fixed initialization for determinism.
  int dim = 2;
  int iter = 50; 
  
  smat::Matrix<double> X0(n, dim);
  X0.set(0, 0, 0.1); X0.set(0, 1, 0.1);
  X0.set(1, 0, 0.1); X0.set(1, 1, 0.9);
  X0.set(2, 0, 0.9); X0.set(2, 1, 0.1);
  X0.set(3, 0, 0.9); X0.set(3, 1, 0.9);

  smat::Matrix<double> X = smat::MDS_SMACOF(D, &X0, dim, iter);
  
  // Calculate distances of X
  smat::Matrix<double> D_rec(n, n);
  smat::EuclideanDistanceMatrix(X, D_rec);
  
  // Verify (tolerant checks)
  for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {
      assertFloatEqual(D.get(i, j), D_rec.get(i, j), 0.05); // 0.05 tolerance
    }
  }

  printTestPass("MDS");
}

void test_linear_algebra()
{
  printTestHeader("Linear Algebra");
  // Test Determinant
  smat::Matrix<double> A(3, 3);
  // 1 2 3
  // 4 5 6  (singular det=0) -> make it non-singular: 6->7
  // 7 8 7   
  // det = 1(35-48) - 2(28-42) + 3(32-35) = 1(-13) - 2(-14) + 3(-3) = -13 + 28 - 9 = 6
  A.set(0,0,1); A.set(0,1,2); A.set(0,2,3);
  A.set(1,0,4); A.set(1,1,5); A.set(1,2,6); // 4 5 6
  A.set(2,0,7); A.set(2,1,8); A.set(2,2,7); // 7 8 7
  
  assertFloatEqual(A.determinant(), 6.0);
  
  // Test Inverse
  smat::Matrix<double> Inv = A.inverse();
  smat::Matrix<double> I = A * Inv;
  // Check if I is identity
  for(int i=0; i<3; i++)
      for(int j=0; j<3; j++)
          assertFloatEqual(I.get(i, j), (i==j)?1.0:0.0);

  // Test LU
  smat::Matrix<double> L, U, P;
  A.lu(L, U, P);
  
  smat::Matrix<double> PA = P * A;
  smat::Matrix<double> LU = L * U;
  
  for(int i=0; i<3; i++)
      for(int j=0; j<3; j++)
          assertFloatEqual(PA.get(i, j), LU.get(i, j));

  assertTrue(isLowerTriangular(L, true), "L must be unit lower triangular");
  assertTrue(isUpperTriangular(U), "U must be upper triangular");
          
  printTestPass("Linear Algebra");
}

void test_qr()
{
  printTestHeader("QR Decomposition");
  smat::Matrix<double> A(3, 3);
  A.set(0,0,12); A.set(0,1,-51); A.set(0,2,4);
  A.set(1,0,6); A.set(1,1,167); A.set(1,2,-68);
  A.set(2,0,-4); A.set(2,1,24); A.set(2,2,-41);
  
  smat::Matrix<double> Q, R;
  A.qr(Q, R);
  
  // Verify A = QR
  smat::Matrix<double> QR = Q * R;
  for(int i=0; i<3; i++)
     for(int j=0; j<3; j++)
         assertFloatEqual(A.get(i, j), QR.get(i, j));
         
  // Verify Q is orthogonal
  assertTrue(isOrthogonal(Q), "Q must be orthogonal");
         
  // Verify R is upper triangular
  assertTrue(isUpperTriangular(R), "R must be upper triangular");
  
  printTestPass("QR Decomposition");
}

void test_eigen()
{
  printTestHeader("Eigen Decomposition");
  smat::Matrix<double> A(3, 3);
  // Symmetric Matrix
  // 2 -1 0
  // -1 2 -1
  // 0 -1 2
  A.set(0,0,2); A.set(0,1,-1); A.set(0,2,0);
  A.set(1,0,-1); A.set(1,1,2); A.set(1,2,-1);
  A.set(2,0,0); A.set(2,1,-1); A.set(2,2,2);
  
  smat::Matrix<double> V, D;
  A.eigen(V, D);
  
  // Verify A V = V D
  smat::Matrix<double> AV = A * V;
  smat::Matrix<double> VD = V * D;
  
  for(int i=0; i<3; i++)
     for(int j=0; j<3; j++)
         assertFloatEqual(AV.get(i, j), VD.get(i, j));
         
  // Verify V is orthogonal (eigenvectors are orthonormal)
  assertTrue(isOrthogonal(V), "V must be orthogonal");

  // Verify D is diagonal (matrix of eigenvalues)
  assertTrue(isDiagonal(D), "D must be diagonal");

  // Verify eigenvalues directly using the new eigenvalues() method
  smat::Matrix<double> eigVals = A.eigenvalues();
  assertIntEq(eigVals.rows(), 3);
  assertIntEq(eigVals.columns(), 1);

  // Expected eigenvalues of [2 -1 0; -1 2 -1; 0 -1 2] are 2-sqrt(2), 2, 2+sqrt(2)
  // which are approx 0.5858, 2.0, 3.4142
  std::vector<double> vals = {eigVals.get(0, 0), eigVals.get(1, 0), eigVals.get(2, 0)};
  std::sort(vals.begin(), vals.end());
  assertFloatEqual(vals[0], 2.0 - std::sqrt(2.0));
  assertFloatEqual(vals[1], 2.0);
  assertFloatEqual(vals[2], 2.0 + std::sqrt(2.0));
         
  printTestPass("Eigen Decomposition");
}

void test_svd()
{
  printTestHeader("SVD");
  smat::Matrix<double> A(3, 2);
  // 1 2
  // 3 4
  // 5 6
  A.set(0,0,1); A.set(0,1,2);
  A.set(1,0,3); A.set(1,1,4);
  A.set(2,0,5); A.set(2,1,6);
  
  smat::Matrix<double> U, S, Vt;
  A.svd(U, S, Vt);
  
  // Verify A = U S Vt
  smat::Matrix<double> US = U * S;
  smat::Matrix<double> USVt = US * Vt;
  
  for(int i=0; i<3; i++)
     for(int j=0; j<2; j++)
         assertFloatEqual(A.get(i, j), USVt.get(i, j));
         
  // Verify U and V are orthogonal
  assertTrue(isOrthogonal(U), "U must be orthogonal-ish (columns are orthonormal)");
  assertTrue(isOrthogonal(Vt.transpose()), "V must be orthogonal");

  // Verify S is diagonal and entries non-negative
  assertTrue(isDiagonal(S), "S must be diagonal");
  for(int i=0; i<2; i++) {
      assertTrue(S.get(i, i) >= -eps, "Singular values must be non-negative");
  }
         
  printTestPass("SVD");
}

void test_rank()
{
  printTestHeader("Rank");
  smat::Matrix<double> A(3, 3);
  // Full rank
  // 1 0 0 
  // 0 1 0
  // 0 0 1
  A.set(0,0,1); A.set(1,1,1); A.set(2,2,1);
  if (A.rank() != 3) { printf("Rank check failed (expected 3, got %d)\n", A.rank()); exit(1); }
  
  // Rank deficient
  // 1 1 1
  // 1 1 1
  // 1 1 1
  // Rank should be 1
  for(int i=0; i<3; i++) for(int j=0; j<3; j++) A.set(i, j, 1.0);
  if (A.rank() != 1) { printf("Rank check failed (expected 1, got %d)\n", A.rank()); exit(1); }
  
  printTestPass("Rank");
}

void test_cpp11_features()
{
  printTestHeader("C++11 features");
  
  // Test Initializer List
  smat::Matrix<double> A = {{1.0, 2.0}, {3.0, 4.0}};
  assertFloatEqual(A.get(0, 0), 1.0);
  assertFloatEqual(A.get(0, 1), 2.0);
  assertFloatEqual(A.get(1, 0), 3.0);
  assertFloatEqual(A.get(1, 1), 4.0);
  
  // Test Move Constructor
  smat::Matrix<double> B = std::move(A);
  assertFloatEqual(B.get(0, 0), 1.0);
  // A should be empty now
  if(A.rows() != 0 || A.columns() != 0) {
      printf("Move constructor failed to clear source.\n");
      exit(1);
  }
  
  // Test Move Assignment
  smat::Matrix<double> C(2, 2);
  C = std::move(B);
  assertFloatEqual(C.get(1, 1), 4.0);
  if(B.rows() != 0) {
      printf("Move assignment failed to clear source.\n");
      exit(1);
  }
  
  printTestPass("C++11 features");
}

void test_solve()
{
  printTestHeader("solve()");
  smat::Matrix<double> A(3, 3);
  // A = [1 2 3; 4 5 6; 7 8 10] (slightly changed from singular [1..9])
  A.set(0,0,1); A.set(0,1,2); A.set(0,2,3);
  A.set(1,0,4); A.set(1,1,5); A.set(1,2,6);
  A.set(2,0,7); A.set(2,1,8); A.set(2,2,10);

  smat::Matrix<double> b(3, 1);
  // Let x = [1; 1; 1], then b = Ax = [6; 15; 25]
  b.set(0,0,6); b.set(1,0,15); b.set(2,0,25);
  
  smat::Matrix<double> x = A.solve(b);
  
  assertFloatEqual(x.get(0,0), 1.0);
  assertFloatEqual(x.get(1,0), 1.0);
  assertFloatEqual(x.get(2,0), 1.0);
  
  printTestPass("solve()");
}

void test_exceptions()
{
  printTestHeader("Exceptions");
  
  smat::Matrix<double> A(2, 2, 1.0);
  
  // Test out_of_range
  try {
    A.get(2, 0);
    assertTrue(false, "Should have thrown out_of_range for get(2, 0)");
  } catch (const std::out_of_range& e) {
    // Success
  }

  // Test dimension mismatch in addition
  smat::Matrix<double> B(3, 3, 1.0);
  try {
    smat::Matrix<double> C = A + B;
    assertTrue(false, "Should have thrown invalid_argument for A + B");
  } catch (const std::invalid_argument& e) {
    // Success
  }

  // Test dot product mismatch
  smat::Matrix<double> D(2, 3);
  try {
    smat::Matrix<double> E = A * D; // 2x2 * 2x3 is OK
    smat::Matrix<double> F = D * A; // 2x3 * 2x2 is NOT OK
    assertTrue(false, "Should have thrown invalid_argument for D * A");
  } catch (const std::invalid_argument& e) {
    // Success
  }

  // Test LU of non-square matrix
  smat::Matrix<double> G(2, 3);
  smat::Matrix<double> L, U, P;
  try {
    G.lu(L, U, P);
    assertTrue(false, "Should have thrown invalid_argument for non-square LU");
  } catch (const std::invalid_argument& e) {
    // Success
  }

  printTestPass("Exceptions");
}

void test_edge_cases()
{
  printTestHeader("Edge Cases");
  
  // 1x1 Matrix
  smat::Matrix<double> A(1, 1, 5.0);
  assertFloatEqual(A.determinant(), 5.0);
  smat::Matrix<double> Inv = A.inverse();
  assertFloatEqual(Inv.get(0, 0), 0.2);
  
  // Singular Matrix
  smat::Matrix<double> S = {{1.0, 1.0}, {1.0, 1.0}};
  assertFloatEqual(S.determinant(), 0.0);
  try {
      S.inverse();
      assertTrue(false, "Should have thrown runtime_error for singular inverse");
  } catch (const std::runtime_error& e) {
      // Success
  }

  // Identity matrix property det(I) = 1
  smat::Matrix<double> I(5, 5, "I");
  assertFloatEqual(I.determinant(), 1.0);
  
  printTestPass("Edge Cases");
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
  test_cpp11_features();
  test_solve();
  test_exceptions();
  test_edge_cases();
  std::cout << "All tests passed!" << std::endl;
  return 0;
}
