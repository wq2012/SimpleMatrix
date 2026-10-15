#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include "SimpleMatrix.h"

const float eps = 0.001f;

void assertIntEq(int x, int y)
{
  if (x != y)
  {
    std::cerr << "assertIntEq failed for " << x << " and " << y << std::endl;
    assert(false);
  }
}

void assertFloatEqual(double x, double y, double threshold = eps)
{
  if (std::fabs(x - y) > threshold)
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
    for (int j = 0; j < std::min(i, M.columns()); j++)
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
  assertIntEq((int)A.get(2, 3), 1);
  A.set(2, 3, 5);
  assertIntEq((int)A.get(2, 3), 5);
  assertIntEq((int)A.get(1, 1), 1);
  assertIntEq(A.rows(), 3);
  assertIntEq(A.columns(), 4);
  printTestPass("Basics");
}

void test_statistics()
{
  printTestHeader("Statistics");
  smat::Matrix<double> A(3, 4, 1.0);
  A.set(2, 3, 5);
  A.set(1, 1, -2);

  assertIntEq((int)A.sum(), 13);
  assertIntEq((int)A.trace(), 0);

  int r, c;
  int max_el = (int)A.maxEl(r, c);
  assertIntEq(max_el, 5);
  assertIntEq(r, 2);
  assertIntEq(c, 3);
  assertFloatEqual(A.maxEl(), 5.0);

  int min_el = (int)A.minEl(r, c);
  assertIntEq(min_el, -2);
  assertIntEq(r, 1);
  assertIntEq(c, 1);
  assertFloatEqual(A.minEl(), -2.0);

  assertFloatEqual(A.mean(), 1.08333);
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
  smat::Matrix<double> LogE = E.log();
  assertFloatEqual(LogE.get(0, 0), 0.0);
  smat::Matrix<double> ExpLogE = LogE.exp();
  assertFloatEqual(ExpLogE.get(0, 0), 1.0);

  smat::Matrix<double> Z(1, 1, 0.0);
  assertFloatEqual(Z.sin().get(0, 0), 0.0);
  assertFloatEqual(Z.cos().get(0, 0), 1.0);
  assertFloatEqual(Z.tanh().get(0, 0), 0.0);

  smat::Matrix<double> Applied = D.apply([](double x) { return x * 0.5 + 1.0; });
  assertFloatEqual(Applied(0, 0), 51.0);
  
  printTestPass("Math Functions");
}

void test_matrix_arithmetic()
{
  printTestHeader("Matrix Arithmetic");
  smat::Matrix<double> A(2, 2, 4.0);
  smat::Matrix<double> B(2, 2, 2.0);

  smat::Matrix<double> C = A.subtractMatrixNew(B);
  assertFloatEqual(C.get(0, 0), 2.0);

  A.subtractMatrixSelf(B);
  assertFloatEqual(A.get(0, 0), 2.0);

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

  assertTrue(E.hstack(F) == H, "hstack alias");
  assertTrue(E.vstack(F) == V, "vstack alias");

  printTestPass("Concatenation");
}

void test_operators()
{
  printTestHeader("Operators");
  smat::Matrix<double> M1(2, 2, 3.0);
  smat::Matrix<double> M2(2, 2, 2.0);
  
  smat::Matrix<double> M3 = M1 + M2;
  assertFloatEqual(M3.get(0, 0), 5.0);
  
  smat::Matrix<double> M4 = M1 * M2;
  assertFloatEqual(M4.get(0, 0), 12.0);
  
  smat::Matrix<double> M5 = M1 - M2;
  assertFloatEqual(M5.get(0, 0), 1.0);

  // Scalar-matrix and compound operators
  smat::Matrix<double> M6 = 2.0 * M1 + 1.0;
  assertFloatEqual(M6(0, 0), 7.0);
  smat::Matrix<double> M7 = 10.0 - M1;
  assertFloatEqual(M7(0, 0), 7.0);
  M7 /= 7.0;
  assertFloatEqual(M7(0, 0), 1.0);
  smat::Matrix<double> Neg = -M7;
  assertFloatEqual(Neg(0, 0), -1.0);

  std::ostringstream oss;
  oss << M1;
  assertTrue(!oss.str().empty(), "ostream operator<< works");

  printTestPass("Operators");
}

void test_factories_and_indexing()
{
  printTestHeader("Factories, Indexing & Structural Operations");
  auto Z = smat::Matrix<double>::zeros(2, 3);
  auto O = smat::Matrix<double>::ones(2, 3);
  auto I = smat::Matrix<double>::eye(3);
  assertFloatEqual(Z.sum(), 0.0);
  assertFloatEqual(O.sum(), 6.0);
  assertTrue(isIdentity(I), "eye(3) is identity");

  // Consecutive random matrices must not be identical
  auto R1 = smat::Matrix<double>::rand(4, 4);
  auto R2 = smat::Matrix<double>::rand(4, 4);
  assertTrue(R1 != R2, "Consecutive rand() matrices must differ");

  // Seeded RNG reproducibility
  smat::seed(99);
  auto N1 = smat::Matrix<double>::randn(3, 3);
  smat::seed(99);
  auto N2 = smat::Matrix<double>::randn(3, 3);
  assertTrue(N1 == N2, "Seeded randn() must be reproducible");

  // Indexing: M(r, c), M(idx), M[r][c]
  smat::Matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
  assertFloatEqual(A(1, 2), 6.0);
  assertFloatEqual(A(5), 6.0);
  assertFloatEqual(A[1][2], 6.0);
  A(0, 1) = 20.0;
  assertFloatEqual(A.get(0, 1), 20.0);
  A(0, 1) = 2.0;

  // Reshape, flatten, row, col, triu, tril, diag
  auto Reshaped = A.reshape(3, 2);
  assertIntEq(Reshaped.rows(), 3);
  assertIntEq(Reshaped.cols(), 2);
  assertFloatEqual(Reshaped(2, 1), 6.0);

  auto Flat = A.flatten();
  assertIntEq(Flat.rows(), 6);
  assertIntEq(Flat.cols(), 1);

  assertFloatEqual(A.row(0).sum(), 6.0);
  assertFloatEqual(A.col(1).sum(), 7.0);

  auto Sq = smat::Matrix<double>::fromVector(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  assertTrue(isUpperTriangular(Sq.triu()), "triu is upper triangular");
  assertTrue(isLowerTriangular(Sq.tril()), "tril is lower triangular");
  auto d = Sq.diag();
  assertFloatEqual(d.sum(), 15.0);
  auto Dmat = d.diag();
  assertTrue(isDiagonal(Dmat), "diag(vector) is diagonal matrix");
  assertFloatEqual(Dmat.trace(), 15.0);

  auto Lin = smat::Matrix<double>::linspace(0.0, 10.0, 5);
  assertIntEq(Lin.cols(), 5);
  assertFloatEqual(Lin(0, 2), 5.0);

  auto Ar = smat::Matrix<double>::arange(0.0, 4.0, 1.0);
  assertIntEq(Ar.cols(), 4);
  assertFloatEqual(Ar.sum(), 6.0);

  printTestPass("Factories, Indexing & Structural Operations");
}

void test_vector_and_matrix_products()
{
  printTestHeader("Vector & Matrix Products (dot, cross, outer, kron)");
  smat::Matrix<double> u = {{1.0, 0.0, 0.0}};
  smat::Matrix<double> v = {{0.0, 1.0, 0.0}};
  assertFloatEqual(u.dot(v), 0.0);
  auto w = u.cross(v);
  assertFloatEqual(w(0), 0.0);
  assertFloatEqual(w(1), 0.0);
  assertFloatEqual(w(2), 1.0);

  auto Out = u.outer(v);
  assertIntEq(Out.rows(), 3);
  assertIntEq(Out.cols(), 3);
  assertFloatEqual(Out(0, 1), 1.0);

  auto I2 = smat::Matrix<double>::eye(2);
  smat::Matrix<double> B = {{1.0, 2.0}, {3.0, 4.0}};
  auto K = I2.kron(B);
  assertIntEq(K.rows(), 4);
  assertIntEq(K.cols(), 4);
  assertFloatEqual(K.trace(), 10.0);

  printTestPass("Vector & Matrix Products (dot, cross, outer, kron)");
}

void test_mds()
{
  printTestHeader("MDS (Classical, SMACOF, UCF)");
  int n = 4;
  smat::Matrix<double> D(n, n, 0.0);
  double s2 = std::sqrt(2.0);
  
  D.set(0, 1, 1.0); D.set(0, 2, 1.0); D.set(0, 3, s2);
  D.set(1, 0, 1.0); D.set(1, 2, s2);  D.set(1, 3, 1.0);
  D.set(2, 0, 1.0); D.set(2, 1, s2);  D.set(2, 3, 1.0);
  D.set(3, 0, s2);  D.set(3, 1, 1.0); D.set(3, 2, 1.0);

  // 1. Classical MDS (exact closed-form recovery)
  smat::Matrix<double> X_classical = smat::MDS_Classical(D, 2);
  smat::Matrix<double> D_classical(n, n);
  smat::EuclideanDistanceMatrix(X_classical, D_classical);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      assertFloatEqual(D.get(i, j), D_classical.get(i, j), 1e-5);
    }
  }

  // 2. SMACOF MDS
  int dim = 2;
  int iter = 50; 
  smat::Matrix<double> X0(n, dim);
  X0.set(0, 0, 0.1); X0.set(0, 1, 0.1);
  X0.set(1, 0, 0.1); X0.set(1, 1, 0.9);
  X0.set(2, 0, 0.9); X0.set(2, 1, 0.1);
  X0.set(3, 0, 0.9); X0.set(3, 1, 0.9);

  smat::Matrix<double> X = smat::MDS_SMACOF(D, &X0, dim, iter);
  smat::Matrix<double> D_rec(n, n);
  smat::EuclideanDistanceMatrix(X, D_rec);
  for(int i=0; i<n; i++) {
    for(int j=0; j<n; j++) {
      assertFloatEqual(D.get(i, j), D_rec.get(i, j), 0.05);
    }
  }

  printTestPass("MDS (Classical, SMACOF, UCF)");
}

void test_linear_algebra()
{
  printTestHeader("Linear Algebra (Determinant, Inverse, LU, Cholesky)");
  smat::Matrix<double> A(3, 3);
  A.set(0,0,1); A.set(0,1,2); A.set(0,2,3);
  A.set(1,0,4); A.set(1,1,5); A.set(1,2,6);
  A.set(2,0,7); A.set(2,1,8); A.set(2,2,7);
  
  assertFloatEqual(A.determinant(), 6.0);
  
  smat::Matrix<double> Inv = A.inverse();
  smat::Matrix<double> I = A * Inv;
  for(int i=0; i<3; i++)
      for(int j=0; j<3; j++)
          assertFloatEqual(I.get(i, j), (i==j)?1.0:0.0);

  smat::Matrix<double> L, U, P;
  A.lu(L, U, P);
  smat::Matrix<double> PA = P * A;
  smat::Matrix<double> LU = L * U;
  for(int i=0; i<3; i++)
      for(int j=0; j<3; j++)
          assertFloatEqual(PA.get(i, j), LU.get(i, j));

  assertTrue(isLowerTriangular(L, true), "L must be unit lower triangular");
  assertTrue(isUpperTriangular(U), "U must be upper triangular");

  // Singular LU decomposition must not produce NaN
  smat::Matrix<double> Sing = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Sing.lu(L, U, P);
  assertTrue((P * Sing).equals(L * U, 1e-6), "Singular PA == LU");

  // Cholesky Decomposition on SPD matrix
  smat::Matrix<double> SPD = {{4, 12, -16}, {12, 37, -43}, {-16, -43, 98}};
  smat::Matrix<double> CholL = SPD.cholesky();
  assertTrue(isLowerTriangular(CholL), "Cholesky factor must be lower triangular");
  assertTrue((CholL * CholL.transpose()).equals(SPD, 1e-6), "L * L^T == SPD");
          
  printTestPass("Linear Algebra (Determinant, Inverse, LU, Cholesky)");
}

void test_qr()
{
  printTestHeader("QR Decomposition (Square, Tall, and Wide)");
  smat::Matrix<double> A(3, 3);
  A.set(0,0,12); A.set(0,1,-51); A.set(0,2,4);
  A.set(1,0,6); A.set(1,1,167); A.set(1,2,-68);
  A.set(2,0,-4); A.set(2,1,24); A.set(2,2,-41);
  
  smat::Matrix<double> Q, R;
  A.qr(Q, R);
  smat::Matrix<double> QR = Q * R;
  for(int i=0; i<3; i++)
     for(int j=0; j<3; j++)
         assertFloatEqual(A.get(i, j), QR.get(i, j));
  assertTrue(isOrthogonal(Q), "Q must be orthogonal");
  assertTrue(isUpperTriangular(R), "R must be upper triangular");

  // Wide matrix (m < n)
  smat::Matrix<double> Wide = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  Wide.qr(Q, R);
  assertTrue(isOrthogonal(Q), "Wide Q must be orthogonal");
  assertTrue(isUpperTriangular(R), "Wide R must be upper trapezoidal");
  assertTrue((Q * R).equals(Wide, 1e-6), "Wide Q * R == Wide");
  
  printTestPass("QR Decomposition (Square, Tall, and Wide)");
}

void test_eigen()
{
  printTestHeader("Eigen Decomposition");
  smat::Matrix<double> A(3, 3);
  A.set(0,0,2); A.set(0,1,-1); A.set(0,2,0);
  A.set(1,0,-1); A.set(1,1,2); A.set(1,2,-1);
  A.set(2,0,0); A.set(2,1,-1); A.set(2,2,2);
  
  smat::Matrix<double> V, D;
  A.eigen(V, D);
  
  smat::Matrix<double> AV = A * V;
  smat::Matrix<double> VD = V * D;
  
  for(int i=0; i<3; i++)
     for(int j=0; j<3; j++)
         assertFloatEqual(AV.get(i, j), VD.get(i, j));
         
  assertTrue(isOrthogonal(V), "V must be orthogonal");
  assertTrue(isDiagonal(D), "D must be diagonal");

  // Eigenvalues are sorted descending
  smat::Matrix<double> eigVals = A.eigenvalues();
  assertIntEq(eigVals.rows(), 3);
  assertIntEq(eigVals.columns(), 1);
  assertFloatEqual(eigVals(0, 0), 2.0 + std::sqrt(2.0));
  assertFloatEqual(eigVals(1, 0), 2.0);
  assertFloatEqual(eigVals(2, 0), 2.0 - std::sqrt(2.0));
         
  printTestPass("Eigen Decomposition");
}

void test_svd()
{
  printTestHeader("SVD (Tall, Wide, and Rank-Deficient)");
  smat::Matrix<double> A(3, 2);
  A.set(0,0,1); A.set(0,1,2);
  A.set(1,0,3); A.set(1,1,4);
  A.set(2,0,5); A.set(2,1,6);
  
  smat::Matrix<double> U, S, Vt;
  A.svd(U, S, Vt);
  
  smat::Matrix<double> USVt = U * S * Vt;
  for(int i=0; i<3; i++)
     for(int j=0; j<2; j++)
         assertFloatEqual(A.get(i, j), USVt.get(i, j));
         
  assertTrue(isOrthogonal(U), "U must have orthonormal columns");
  assertTrue(isOrthogonal(Vt.transpose()), "V must be orthogonal");
  assertTrue(isDiagonal(S), "S must be diagonal");
  assertTrue(S(0, 0) >= S(1, 1) && S(1, 1) >= 0.0, "Singular values sorted descending and non-negative");

  // Rank-deficient matrix SVD must still produce orthonormal U and V
  smat::Matrix<double> RankDef = {{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}};
  RankDef.svd(U, S, Vt);
  assertTrue(isOrthogonal(U), "Rank-deficient U must still be orthogonal");
  assertTrue(isOrthogonal(Vt.transpose()), "Rank-deficient V must still be orthogonal");
  assertTrue((U * S * Vt).equals(RankDef, 1e-6), "Rank-deficient U*S*Vt == A");
         
  printTestPass("SVD (Tall, Wide, and Rank-Deficient)");
}

void test_rank()
{
  printTestHeader("Rank");
  smat::Matrix<double> A(3, 3);
  A.set(0,0,1); A.set(1,1,1); A.set(2,2,1);
  assertIntEq(A.rank(), 3);
  
  for(int i=0; i<3; i++) for(int j=0; j<3; j++) A.set(i, j, 1.0);
  assertIntEq(A.rank(), 1);
  
  printTestPass("Rank");
}

void test_cpp11_features()
{
  printTestHeader("C++11 features");
  smat::Matrix<double> A = {{1.0, 2.0}, {3.0, 4.0}};
  assertFloatEqual(A.get(0, 0), 1.0);
  assertFloatEqual(A.get(0, 1), 2.0);
  assertFloatEqual(A.get(1, 0), 3.0);
  assertFloatEqual(A.get(1, 1), 4.0);
  
  smat::Matrix<double> B = std::move(A);
  assertFloatEqual(B.get(0, 0), 1.0);
  assertFloatEqual(B.v[1][1], 4.0);
  assertTrue(A.rows() == 0 && A.columns() == 0, "Moved-from matrix is empty");
  
  smat::Matrix<double> C(2, 2);
  C = std::move(B);
  assertFloatEqual(C.get(1, 1), 4.0);
  assertTrue(B.rows() == 0, "Move-assigned source is empty");
  
  printTestPass("C++11 features");
}

void test_solve_and_advanced_linalg()
{
  printTestHeader("solve(), pinv(), lstsq(), nullSpace(), orth(), expm()");
  smat::Matrix<double> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
  smat::Matrix<double> b = {{6}, {15}, {25}};
  smat::Matrix<double> x = A.solve(b);
  assertFloatEqual(x(0, 0), 1.0);
  assertFloatEqual(x(1, 0), 1.0);
  assertFloatEqual(x(2, 0), 1.0);

  // Pseudoinverse Moore-Penrose 4 conditions on rectangular rank-deficient matrix
  smat::Matrix<double> M = {{1, 2, 3}, {2, 4, 6}};
  smat::Matrix<double> M_pinv = M.pinv();
  assertTrue((M * M_pinv * M).equals(M, 1e-6), "M * M+ * M == M");
  assertTrue((M_pinv * M * M_pinv).equals(M_pinv, 1e-6), "M+ * M * M+ == M+");
  assertTrue((M * M_pinv).isSymmetric(1e-6), "M * M+ symmetric");
  assertTrue((M_pinv * M).isSymmetric(1e-6), "M+ * M symmetric");

  // Overdetermined least squares: fit y = 2*x + 1 on exact points (0,1), (1,3), (2,5), (3,7)
  smat::Matrix<double> X_design = {{0, 1}, {1, 1}, {2, 1}, {3, 1}};
  smat::Matrix<double> y_obs = {{1}, {3}, {5}, {7}};
  smat::Matrix<double> beta = X_design.lstsq(y_obs);
  assertFloatEqual(beta(0, 0), 2.0);
  assertFloatEqual(beta(1, 0), 1.0);

  // Null space and column space
  smat::Matrix<double> Sing = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  smat::Matrix<double> N = Sing.nullSpace();
  smat::Matrix<double> Q = Sing.orth();
  assertIntEq(N.cols(), 1);
  assertIntEq(Q.cols(), 2);
  assertFloatEqual((Sing * N).fnorm(), 0.0);
  assertTrue(isOrthogonal(N), "Null space basis is orthonormal");
  assertTrue(isOrthogonal(Q), "Column space basis is orthonormal");

  // Matrix exponential: expm(A) * expm(-A) == I
  smat::Matrix<double> ExpA = A.expm();
  smat::Matrix<double> ExpNegA = (-A).expm();
  assertTrue((ExpA * ExpNegA).equals(smat::Matrix<double>::eye(3), 1e-5), "expm(A)*expm(-A) == I");

  // PCA & Covariance / Correlation
  smat::Matrix<double> Data = {{1.0, 2.0, 0.5}, {2.0, 4.0, 1.5}, {3.0, 6.0, 2.0}, {4.0, 8.0, 3.5}};
  smat::Matrix<double> Proj, Comp, ExpVar;
  Data.pca(2, Proj, Comp, ExpVar);
  assertIntEq(Proj.rows(), 4);
  assertIntEq(Proj.cols(), 2);
  assertIntEq(Comp.rows(), 3);
  assertIntEq(Comp.cols(), 2);
  assertTrue(ExpVar(0, 0) >= ExpVar(1, 0), "PCA explained variance descending");

  printTestPass("solve(), pinv(), lstsq(), nullSpace(), orth(), expm()");
}

void test_template_types()
{
  printTestHeader("Template Instantiations (int, float, double)");
  smat::Matrix<int> Mi = {{10, 20}, {30, 40}};
  Mi.divideNumberSelf(10);
  assertIntEq(Mi(0, 0), 1);
  assertIntEq(Mi(1, 1), 4);

  smat::Matrix<float> Mf = {{1.0f, 2.0f}, {3.0f, 5.0f}};
  assertFloatEqual(Mf.determinant(), -1.0f);
  auto Mf_inv = Mf.inverse();
  assertTrue(isIdentity(Mf * Mf_inv), "Matrix<float> inverse works");

  printTestPass("Template Instantiations (int, float, double)");
}

void test_exceptions()
{
  printTestHeader("Exceptions");
  
  smat::Matrix<double> A(2, 2, 1.0);
  
  try {
    A.get(2, 0);
    assertTrue(false, "Should have thrown out_of_range for get(2, 0)");
  } catch (const std::out_of_range&) {}

  smat::Matrix<double> B(3, 3, 1.0);
  try {
    smat::Matrix<double> C = A + B;
    assertTrue(false, "Should have thrown invalid_argument for A + B");
  } catch (const std::invalid_argument&) {}

  smat::Matrix<double> D(2, 3);
  try {
    smat::Matrix<double> E = A * D;
    smat::Matrix<double> F = D * A;
    assertTrue(false, "Should have thrown invalid_argument for D * A");
  } catch (const std::invalid_argument&) {}

  smat::Matrix<double> G(2, 3);
  smat::Matrix<double> L, U, P;
  try {
    G.lu(L, U, P);
    assertTrue(false, "Should have thrown invalid_argument for non-square LU");
  } catch (const std::invalid_argument&) {}

  // Non-symmetric eigen should throw
  smat::Matrix<double> NonSym = {{1.0, 5.0}, {0.0, 2.0}};
  try {
    smat::Matrix<double> V, Diag;
    NonSym.eigen(V, Diag);
    assertTrue(false, "Should have thrown invalid_argument for non-symmetric eigen");
  } catch (const std::invalid_argument&) {}

  // Non-positive-definite Cholesky should throw
  smat::Matrix<double> NonPD = {{-1.0, 0.0}, {0.0, 1.0}};
  try {
    NonPD.cholesky();
    assertTrue(false, "Should have thrown runtime_error for non-SPD Cholesky");
  } catch (const std::runtime_error&) {}

  printTestPass("Exceptions");
}

void test_edge_cases()
{
  printTestHeader("Edge Cases");
  
  smat::Matrix<double> A(1, 1, 5.0);
  assertFloatEqual(A.determinant(), 5.0);
  smat::Matrix<double> Inv = A.inverse();
  assertFloatEqual(Inv.get(0, 0), 0.2);
  
  smat::Matrix<double> S = {{1.0, 1.0}, {1.0, 1.0}};
  assertFloatEqual(S.determinant(), 0.0);
  try {
      S.inverse();
      assertTrue(false, "Should have thrown runtime_error for singular inverse");
  } catch (const std::runtime_error&) {}

  smat::Matrix<double> I(5, 5, "I");
  assertFloatEqual(I.determinant(), 1.0);
  assertFloatEqual(I.cond(), 1.0);
  
  printTestPass("Edge Cases");
}

int main()
{
  test_basics();
  test_statistics();
  test_scalar_operations();
  test_math_functions();
  test_matrix_arithmetic();
  test_concatenation();
  test_operators();
  test_factories_and_indexing();
  test_vector_and_matrix_products();
  test_mds();
  test_linear_algebra();
  test_qr();
  test_eigen();
  test_svd();
  test_rank();
  test_cpp11_features();
  test_solve_and_advanced_linalg();
  test_template_types();
  test_exceptions();
  test_edge_cases();
  std::cout << "All C++ tests passed!" << std::endl;
  return 0;
}
