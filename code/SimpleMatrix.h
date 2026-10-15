/**
 * @file SimpleMatrix.h
 * @brief Lightweight, Single-Header C++ Matrix & Linear Algebra Library.
 * @author Quan Wang
 * @copyright Copyright (C) 2013-2026 Quan Wang
 * @license FreeBSD (2-Clause BSD) License
 *
 * This is the standalone C++ header of SimpleMatrix.
 * (For pure C projects, include "SimpleMatrixC.h" instead.)
 *
 * Features:
 *   - Template class smat::Matrix<T> with contiguous RAII storage (std::vector<T>)
 *     plus legacy-compatible 2D row pointers (M.v[i][j]) and M(i, j) indexing.
 *   - Full operator overloading (+, -, *, +=, -=, *=, /=, ==, !=, <<).
 *   - Slicing, reshaping, concatenation, Kronecker/outer/cross/dot products.
 *   - Complete Linear Algebra suite:
 *       * LU decomposition with partial pivoting (PA = LU)
 *       * Cholesky decomposition (A = L * L^T)
 *       * QR decomposition via Householder reflections (A = Q * R)
 *       * Symmetric Eigenvalue decomposition via Cyclic Jacobi (A = V * D * V^T)
 *       * Singular Value Decomposition (SVD, A = U * S * V^T)
 *       * Determinant, Rank, Condition Number, Matrix Norms (1, 2, Inf, Frobenius, p)
 *       * Matrix Inverse & Moore-Penrose Pseudoinverse (pinv)
 *       * Linear System Solver (Ax = B) & Least Squares Solver (lstsq)
 *       * Orthonormal Basis (orth) & Null Space (nullSpace)
 *       * Matrix Exponential (expm via Scaling and Squaring Pade Approximation)
 *       * Covariance (cov), Pearson Correlation (corrcoef), and PCA
 *       * Multidimensional Scaling (MDS_Classical, MDS_SMACOF, MDS_UCF)
 */

#ifndef SIMPLE_MATRIX_H
#define SIMPLE_MATRIX_H

#ifndef __cplusplus
#error "SimpleMatrix.h is for C++ projects. For pure C projects, please include SimpleMatrixC.h instead."
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <initializer_list>
#include <algorithm>
#include <random>
#include <functional>

#ifndef MAX_LINE_LENGTH
#define MAX_LINE_LENGTH 100000
#endif

#ifndef EPSILON
#define EPSILON 0.0000001
#endif

namespace smat {

inline std::mt19937& globalRng() {
  static std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
  return rng;
}

inline void seed(unsigned int s) {
  globalRng().seed(s);
}

/**********************************************
 * Declaration part
 **********************************************/

template <class T>
class Matrix {
private:
  int rows_;
  int columns_;
  std::vector<T> data_;

  void syncRowPointers() {
    v.resize(rows_);
    for (int i = 0; i < rows_; ++i) {
      v[i] = data_.data() + static_cast<size_t>(i) * static_cast<size_t>(columns_);
    }
  }

public:
  /**
   * Row pointer table for backward-compatible 2D indexing: M.v[r][c].
   * Points into contiguous storage data_.
   */
  std::vector<T*> v;

  static void seedRng(unsigned int s) {
    seed(s);
  }

  // Constructors
  Matrix();                                       // default empty constructor (0x0)
  Matrix(int rows, int columns);                  // zero-initialized constructor
  Matrix(int rows, int columns, T value);         // value-initialized constructor
  Matrix(int rows, int columns, std::string type);// special matrix ("I", "eye", "zeros", "ones", "rand", "randn", "rand_int", "randperm")
  Matrix(const Matrix &other);                    // copy constructor
  Matrix(Matrix &&other) noexcept;                // move constructor
  Matrix(std::initializer_list<std::initializer_list<T>> list); // 2D initializer list
  explicit Matrix(const char *filename);          // load from text/CSV file
  explicit Matrix(const std::string &filename);   // load from text/CSV file

  // Static factory methods
  static Matrix<T> zeros(int rows, int columns);
  static Matrix<T> ones(int rows, int columns);
  static Matrix<T> eye(int n);
  static Matrix<T> eye(int rows, int columns);
  static Matrix<T> identity(int n);
  static Matrix<T> rand(int rows, int columns);
  static Matrix<T> randn(int rows, int columns);
  static Matrix<T> randInt(int rows, int columns, int low = 0, int high = RAND_MAX);
  static Matrix<T> randperm(int rows, int columns = 1);
  static Matrix<T> diag(const Matrix<T> &vec_or_mat);
  static Matrix<T> diag(const std::vector<T> &vec);
  static Matrix<T> linspace(T start, T end, int n);
  static Matrix<T> arange(T start, T stop, T step = static_cast<T>(1));
  static Matrix<T> fromVector(int rows, int columns, const std::vector<T> &vec);
  static Matrix<T> fromArray(int rows, int columns, const T *arr);

  // Assignment operators
  Matrix<T> &operator=(const Matrix<T> &other);
  Matrix<T> &operator=(Matrix<T> &&other) noexcept;

  // Destructor
  ~Matrix();

  // Element access & properties
  void set(int r, int c, T value);
  T get(int r, int c) const;
  T& operator()(int r, int c);
  const T& operator()(int r, int c) const;
  T& operator()(int linear_idx);
  const T& operator()(int linear_idx) const;
  T* operator[](int r) { return v[r]; }
  const T* operator[](int r) const { return v[r]; }

  int rows() const;
  int columns() const;
  int cols() const { return columns_; }
  int size() const { return rows_ * columns_; }
  bool empty() const { return rows_ == 0 || columns_ == 0; }
  bool isSquare() const { return rows_ == columns_ && rows_ > 0; }
  bool isVector() const { return (rows_ == 1 && columns_ >= 1) || (columns_ == 1 && rows_ >= 1); }
  bool isSymmetric(double tol = 1e-9) const;
  bool equals(const Matrix<T> &other, double tol = 1e-9) const;

  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }
  const std::vector<T>& toVector() const { return data_; }

  void fill(T value);
  void print() const;
  Matrix<T> copy() const;
  void saveTxt(const char *filename) const;
  void saveTxt(const std::string &filename) const { saveTxt(filename.c_str()); }

  // Shape, Slicing & Structural Operations
  Matrix<T> transpose() const;
  Matrix<T> T_mat() const { return transpose(); }
  Matrix<T> reshape(int new_rows, int new_cols) const;
  Matrix<T> flatten() const;
  Matrix<T> sub(int r1, int r2, int c1, int c2) const; // inclusive [r1, r2] x [c1, c2]
  Matrix<T> row(int r) const;
  Matrix<T> col(int c) const;
  void setRow(int r, const Matrix<T> &rowVec);
  void setCol(int c, const Matrix<T> &colVec);
  void swapRows(int r1, int r2);
  void swapCols(int c1, int c2);
  Matrix<T> diag() const;
  Matrix<T> triu(int k = 0) const;
  Matrix<T> tril(int k = 0) const;
  Matrix<T> concatenateRight(const Matrix<T> &A) const;
  Matrix<T> concatenateBottom(const Matrix<T> &A) const;
  Matrix<T> hstack(const Matrix<T> &A) const { return concatenateRight(A); }
  Matrix<T> vstack(const Matrix<T> &A) const { return concatenateBottom(A); }

  // Reductions, Norms & Statistics
  T trace() const;
  double fnorm() const;
  double pnorm(double p) const;
  double norm1() const;
  double normInf() const;
  double norm2() const;
  double cond() const;
  T maxEl(int &r, int &c) const;
  T maxEl() const;
  T minEl(int &r, int &c) const;
  T minEl() const;
  double mean() const;
  T sum() const;
  double var(bool sample = true) const;
  double std(bool sample = true) const;
  Matrix<T> sumAxis(int axis) const;
  Matrix<double> meanAxis(int axis) const;
  Matrix<T> minAxis(int axis) const;
  Matrix<T> maxAxis(int axis) const;
  Matrix<double> cov() const;
  Matrix<double> corrcoef() const;

  // In-place scalar operations
  void addNumberSelf(T value);
  void subtractNumberSelf(T value);
  void multiplyNumberSelf(T value);
  void divideNumberSelf(T value);

  // Element-wise mathematical functions
  Matrix<T> abs() const;
  Matrix<T> exp() const;
  Matrix<T> log() const;
  Matrix<T> sqrt() const;
  Matrix<T> power(double p) const;
  Matrix<T> sin() const;
  Matrix<T> cos() const;
  Matrix<T> tanh() const;
  Matrix<T> apply(std::function<T(T)> fn) const;

  // In-place element-wise matrix operations
  void addMatrixSelf(const Matrix<T> &A);
  void subtractMatrixSelf(const Matrix<T> &A);
  void dotMultiplyMatrixSelf(const Matrix<T> &A);
  void dotDivideMatrixSelf(const Matrix<T> &A);

  // New-matrix arithmetic & products
  Matrix<T> addMatrixNew(const Matrix<T> &A) const;
  Matrix<T> subtractMatrixNew(const Matrix<T> &A) const;
  Matrix<T> dotMultiplyMatrixNew(const Matrix<T> &A) const;
  Matrix<T> dotDivideMatrixNew(const Matrix<T> &A) const;
  Matrix<T> multiplyMatrixNew(const Matrix<T> &A) const;
  T dot(const Matrix<T> &other) const;
  Matrix<T> outer(const Matrix<T> &other) const;
  Matrix<T> cross(const Matrix<T> &other) const;
  Matrix<T> kron(const Matrix<T> &other) const;

  // Linear Algebra Decompositions & Solvers
  void lu(Matrix<T> &L, Matrix<T> &U, Matrix<T> &P) const;
  Matrix<T> cholesky() const;
  void cholesky(Matrix<T> &L) const;
  void qr(Matrix<T> &Q, Matrix<T> &R) const;
  void eigen(Matrix<T> &V, Matrix<T> &D) const;
  Matrix<T> eigenvalues() const;
  void svd(Matrix<T> &U, Matrix<T> &S, Matrix<T> &Vt) const;
  T determinant() const;
  T det() const { return determinant(); }
  int rank(double tol = -1.0) const;
  Matrix<T> inverse() const;
  Matrix<T> inv() const { return inverse(); }
  Matrix<T> pinv(double tol = -1.0) const;
  Matrix<T> solve(const Matrix<T> &B) const;
  Matrix<T> lstsq(const Matrix<T> &B) const;
  Matrix<T> orth(double tol = -1.0) const;
  Matrix<T> nullSpace(double tol = -1.0) const;
  Matrix<T> expm() const;
  void pca(int n_components, Matrix<T> &projected, Matrix<T> &components, Matrix<T> &explained_variance) const;

  // Operator overloading
  Matrix<T> operator+(const Matrix<T> &other) const;
  Matrix<T> operator-(const Matrix<T> &other) const;
  Matrix<T> operator*(const Matrix<T> &other) const;
  Matrix<T>& operator+=(const Matrix<T> &other);
  Matrix<T>& operator-=(const Matrix<T> &other);
  Matrix<T>& operator*=(const Matrix<T> &other);

  Matrix<T> operator-() const;
  Matrix<T> operator+(T scalar) const;
  Matrix<T> operator-(T scalar) const;
  Matrix<T> operator*(T scalar) const;
  Matrix<T> operator/(T scalar) const;
  Matrix<T>& operator+=(T scalar);
  Matrix<T>& operator-=(T scalar);
  Matrix<T>& operator*=(T scalar);
  Matrix<T>& operator/=(T scalar);

  bool operator==(const Matrix<T> &other) const;
  bool operator!=(const Matrix<T> &other) const;
};

typedef Matrix<double> Matd;
typedef Matrix<float>  Matf;
typedef Matrix<int>    Mati;

/**********************************************
 * Utilities part
 **********************************************/

template <class T>
inline T min(T v1, T v2) { return (v1 < v2) ? v1 : v2; }

template <class T>
inline T max(T v1, T v2) { return (v1 > v2) ? v1 : v2; }

template <class T>
inline void swap(T &v1, T &v2) {
  T v3 = v1;
  v1 = v2;
  v2 = v3;
}

template <class T>
inline double sign(T v) {
  if (v > 0) return 1.0;
  else if (v < 0) return -1.0;
  else return 0.0;
}

namespace detail {

/* Internal helper: cast Matrix<T> to Matrix<double> */
template <class T>
inline Matrix<double> toDoubleMatrix(const Matrix<T> &M) {
  if (M.empty()) return Matrix<double>();
  Matrix<double> D(M.rows(), M.columns());
  const T *src = M.data();
  double *dst = D.data();
  int total = M.size();
  for (int i = 0; i < total; ++i) {
    dst[i] = static_cast<double>(src[i]);
  }
  return D;
}

/* Internal helper: cast Matrix<double> to Matrix<T> */
template <class T>
inline Matrix<T> fromDoubleMatrix(const Matrix<double> &D) {
  if (D.empty()) return Matrix<T>();
  Matrix<T> M(D.rows(), D.columns());
  const double *src = D.data();
  T *dst = M.data();
  int total = D.size();
  for (int i = 0; i < total; ++i) {
    dst[i] = static_cast<T>(src[i]);
  }
  return M;
}

/* Internal helper: LU decomposition with partial pivoting returning swap count */
inline int luDecomposeDouble(const Matrix<double> &A_in, Matrix<double> &LU, std::vector<int> &piv) {
  int n = A_in.rows();
  LU = A_in.copy();
  piv.resize(n);
  for (int i = 0; i < n; ++i) piv[i] = i;
  int swap_count = 0;

  for (int k = 0; k < n; ++k) {
    double max_val = std::fabs(LU.v[k][k]);
    int max_row = k;
    for (int i = k + 1; i < n; ++i) {
      double val = std::fabs(LU.v[i][k]);
      if (val > max_val) {
        max_val = val;
        max_row = i;
      }
    }
    if (max_row != k) {
      LU.swapRows(k, max_row);
      std::swap(piv[k], piv[max_row]);
      swap_count++;
    }
    if (std::fabs(LU.v[k][k]) > 1e-15) {
      for (int i = k + 1; i < n; ++i) {
        LU.v[i][k] /= LU.v[k][k];
        double mult = LU.v[i][k];
        for (int j = k + 1; j < n; ++j) {
          LU.v[i][j] -= mult * LU.v[k][j];
        }
      }
    }
  }
  return swap_count;
}

/* Internal helper: Symmetric Eigenvalue Decomposition via Cyclic Jacobi */
inline void eigenSymmetricDouble(const Matrix<double> &A, Matrix<double> &V, Matrix<double> &D) {
  int n = A.rows();
  V = Matrix<double>::eye(n);
  Matrix<double> W = A.copy();
  const int max_sweeps = 100;

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    double off_diag_norm = 0.0;
    for (int p = 0; p < n - 1; ++p) {
      for (int q = p + 1; q < n; ++q) {
        off_diag_norm += W.v[p][q] * W.v[p][q];
      }
    }
    if (std::sqrt(off_diag_norm) < 1e-15) break;

    for (int p = 0; p < n - 1; ++p) {
      for (int q = p + 1; q < n; ++q) {
        double apq = W.v[p][q];
        if (std::fabs(apq) < 1e-15) continue;
        double app = W.v[p][p];
        double aqq = W.v[q][q];
        double tau = (aqq - app) / (2.0 * apq);
        double t = (tau >= 0.0)
                   ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                   : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
        double c = 1.0 / std::sqrt(1.0 + t * t);
        double s = t * c;

        W.v[p][p] = app - t * apq;
        W.v[q][q] = aqq + t * apq;
        W.v[p][q] = 0.0;
        W.v[q][p] = 0.0;

        for (int r = 0; r < n; ++r) {
          if (r != p && r != q) {
            double arp = W.v[r][p];
            double arq = W.v[r][q];
            W.v[r][p] = c * arp - s * arq;
            W.v[p][r] = W.v[r][p];
            W.v[r][q] = s * arp + c * arq;
            W.v[q][r] = W.v[r][q];
          }
        }
        for (int r = 0; r < n; ++r) {
          double vrp = V.v[r][p];
          double vrq = V.v[r][q];
          V.v[r][p] = c * vrp - s * vrq;
          V.v[r][q] = s * vrp + c * vrq;
        }
      }
    }
  }

  // Sort eigenvalues descending
  for (int i = 0; i < n - 1; ++i) {
    int max_idx = i;
    double max_val = W.v[i][i];
    for (int j = i + 1; j < n; ++j) {
      if (W.v[j][j] > max_val) {
        max_val = W.v[j][j];
        max_idx = j;
      }
    }
    if (max_idx != i) {
      std::swap(W.v[i][i], W.v[max_idx][max_idx]);
      V.swapCols(i, max_idx);
    }
  }

  D = Matrix<double>::zeros(n, n);
  for (int i = 0; i < n; ++i) {
    D.v[i][i] = W.v[i][i];
  }
}

/* Internal helper: QR Decomposition via Householder Reflections */
inline void qrDouble(const Matrix<double> &A, Matrix<double> &Q, Matrix<double> &R) {
  int m = A.rows();
  int n = A.columns();
  Q = Matrix<double>::eye(m);
  R = A.copy();
  int k_max = std::min(m, n);
  std::vector<double> u(m);

  for (int k = 0; k < k_max; ++k) {
    double norm_x = 0.0;
    for (int i = k; i < m; ++i) {
      norm_x += R.v[i][k] * R.v[i][k];
    }
    norm_x = std::sqrt(norm_x);
    if (norm_x < 1e-15) continue;

    double s = (R.v[k][k] >= 0.0) ? 1.0 : -1.0;
    double u1 = R.v[k][k] + s * norm_x;
    u[k] = 1.0;
    double u_norm_sq = 1.0;
    for (int i = k + 1; i < m; ++i) {
      u[i] = R.v[i][k] / u1;
      u_norm_sq += u[i] * u[i];
    }
    double beta = 2.0 / u_norm_sq;

    // Update R(k:m-1, k:n-1)
    for (int j = k; j < n; ++j) {
      double dot = 0.0;
      for (int i = k; i < m; ++i) dot += u[i] * R.v[i][j];
      dot *= beta;
      for (int i = k; i < m; ++i) R.v[i][j] -= dot * u[i];
    }
    // Update Q(0:m-1, k:m-1)
    for (int i = 0; i < m; ++i) {
      double dot = 0.0;
      for (int j = k; j < m; ++j) dot += Q.v[i][j] * u[j];
      dot *= beta;
      for (int j = k; j < m; ++j) Q.v[i][j] -= dot * u[j];
    }
  }

  // Enforce positive diagonal in R for uniqueness
  for (int k = 0; k < k_max; ++k) {
    if (R.v[k][k] < 0.0) {
      for (int j = k; j < n; ++j) R.v[k][j] = -R.v[k][j];
      for (int i = 0; i < m; ++i) Q.v[i][k] = -Q.v[i][k];
    }
  }
}

/* Internal helper: Thin/Full SVD via One-Sided Jacobi Orthogonalization + QR completion */
inline void svdDouble(const Matrix<double> &A, Matrix<double> &U, Matrix<double> &S, Matrix<double> &Vt) {
  int m = A.rows();
  int n = A.columns();

  if (m < n) {
    Matrix<double> At = A.transpose();
    Matrix<double> Ut_sub, S_sub, Vt_sub;
    svdDouble(At, Ut_sub, S_sub, Vt_sub);
    U = Vt_sub.transpose();
    S = S_sub.transpose();
    Vt = Ut_sub.transpose();
    return;
  }

  // Here m >= n
  Matrix<double> W = A.copy();
  Matrix<double> V = Matrix<double>::eye(n);
  const int max_sweeps = 100;

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    bool converged = true;
    for (int p = 0; p < n - 1; ++p) {
      for (int q = p + 1; q < n; ++q) {
        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        for (int i = 0; i < m; ++i) {
          double wp = W.v[i][p];
          double wq = W.v[i][q];
          alpha += wp * wp;
          beta  += wq * wq;
          gamma += wp * wq;
        }
        if (std::fabs(gamma) <= 1e-15 * std::sqrt(alpha * beta)) continue;
        converged = false;

        double zeta = (beta - alpha) / (2.0 * gamma);
        double t = (zeta >= 0.0)
                   ? (1.0 / (zeta + std::sqrt(1.0 + zeta * zeta)))
                   : (-1.0 / (-zeta + std::sqrt(1.0 + zeta * zeta)));
        double c = 1.0 / std::sqrt(1.0 + t * t);
        double s = c * t;

        for (int i = 0; i < m; ++i) {
          double wp = W.v[i][p];
          double wq = W.v[i][q];
          W.v[i][p] = c * wp - s * wq;
          W.v[i][q] = s * wp + c * wq;
        }
        for (int i = 0; i < n; ++i) {
          double vp = V.v[i][p];
          double vq = V.v[i][q];
          V.v[i][p] = c * vp - s * vq;
          V.v[i][q] = s * vp + c * vq;
        }
      }
    }
    if (converged) break;
  }

  std::vector<double> sing(n);
  for (int j = 0; j < n; ++j) {
    double col_norm = 0.0;
    for (int i = 0; i < m; ++i) col_norm += W.v[i][j] * W.v[i][j];
    sing[j] = std::sqrt(col_norm);
  }

  // Sort singular values descending
  for (int i = 0; i < n - 1; ++i) {
    int max_idx = i;
    for (int j = i + 1; j < n; ++j) {
      if (sing[j] > sing[max_idx]) max_idx = j;
    }
    if (max_idx != i) {
      std::swap(sing[i], sing[max_idx]);
      W.swapCols(i, max_idx);
      V.swapCols(i, max_idx);
    }
  }

  U = Matrix<double>::zeros(m, n);
  S = Matrix<double>::zeros(n, n);
  int rank_cols = 0;
  for (int j = 0; j < n; ++j) {
    S.v[j][j] = sing[j];
    if (sing[j] > 1e-14) {
      for (int i = 0; i < m; ++i) {
        U.v[i][j] = W.v[i][j] / sing[j];
      }
      rank_cols++;
    }
  }

  // Complete orthonormal columns of U if rank_cols < n
  if (rank_cols < n) {
    Matrix<double> U_aug(m, m, 0.0);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < rank_cols; ++j) U_aug.v[i][j] = U.v[i][j];
      if (i >= rank_cols) U_aug.v[i][i] = 1.0;
    }
    Matrix<double> Q_full, R_full;
    qrDouble(U_aug, Q_full, R_full);
    for (int j = rank_cols; j < n; ++j) {
      for (int i = 0; i < m; ++i) U.v[i][j] = Q_full.v[i][j];
    }
  }

  Vt = V.transpose();
}

} // namespace detail

/**********************************************
 * Implementation part
 **********************************************/

template <class T>
Matrix<T>::Matrix() : rows_(0), columns_(0) {}

template <class T>
Matrix<T>::Matrix(int rows, int columns) {
  if (rows < 1 || columns < 1) {
    throw std::invalid_argument("Invalid construction arguments: rows=" + std::to_string(rows) + ", columns=" + std::to_string(columns));
  }
  rows_ = rows;
  columns_ = columns;
  data_.assign(static_cast<size_t>(rows) * static_cast<size_t>(columns), static_cast<T>(0));
  syncRowPointers();
}

template <class T>
Matrix<T>::Matrix(int rows, int columns, T value) {
  if (rows < 1 || columns < 1) {
    throw std::invalid_argument("Invalid construction arguments: rows=" + std::to_string(rows) + ", columns=" + std::to_string(columns));
  }
  rows_ = rows;
  columns_ = columns;
  data_.assign(static_cast<size_t>(rows) * static_cast<size_t>(columns), value);
  syncRowPointers();
}

template <class T>
Matrix<T>::Matrix(int rows, int columns, std::string type) {
  if (rows < 1 || columns < 1) {
    throw std::invalid_argument("Invalid construction arguments: rows=" + std::to_string(rows) + ", columns=" + std::to_string(columns));
  }
  rows_ = rows;
  columns_ = columns;
  data_.assign(static_cast<size_t>(rows) * static_cast<size_t>(columns), static_cast<T>(0));
  syncRowPointers();

  if (type == "I" || type == "eye" || type == "identity") {
    for (int i = 0; i < std::min(rows_, columns_); i++) {
      v[i][i] = static_cast<T>(1);
    }
  } else if (type == "zeros") {
    // Already zero-initialized
  } else if (type == "ones") {
    std::fill(data_.begin(), data_.end(), static_cast<T>(1));
  } else if (type == "rand") {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    auto &rng = globalRng();
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] = static_cast<T>(dist(rng));
    }
  } else if (type == "randn") {
    std::normal_distribution<double> dist(0.0, 1.0);
    auto &rng = globalRng();
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] = static_cast<T>(dist(rng));
    }
  } else if (type == "rand_int") {
    std::uniform_int_distribution<int> dist(0, RAND_MAX);
    auto &rng = globalRng();
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] = static_cast<T>(dist(rng));
    }
  } else if (type == "randperm") {
    auto &rng = globalRng();
    for (int j = 0; j < columns_; ++j) {
      std::vector<int> col_perm(rows_);
      for (int k = 0; k < rows_; ++k) col_perm[k] = k + 1;
      std::shuffle(col_perm.begin(), col_perm.end(), rng);
      for (int i = 0; i < rows_; ++i) v[i][j] = static_cast<T>(col_perm[i]);
    }
  } else {
    throw std::invalid_argument("Undefined matrix type: " + type);
  }
}

template <class T>
Matrix<T>::Matrix(const Matrix &other)
  : rows_(other.rows_), columns_(other.columns_), data_(other.data_) {
  syncRowPointers();
}

template <class T>
Matrix<T>::Matrix(Matrix &&other) noexcept
  : rows_(other.rows_), columns_(other.columns_), data_(std::move(other.data_)) {
  syncRowPointers();
  other.rows_ = 0;
  other.columns_ = 0;
  other.v.clear();
}

template <class T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list) {
  rows_ = static_cast<int>(list.size());
  if (rows_ == 0) {
    columns_ = 0;
    return;
  }
  columns_ = static_cast<int>(list.begin()->size());
  if (columns_ == 0) {
    rows_ = 0;
    return;
  }
  data_.resize(static_cast<size_t>(rows_) * static_cast<size_t>(columns_));
  syncRowPointers();

  int i = 0;
  for (const auto &row : list) {
    if (static_cast<int>(row.size()) != columns_) {
      throw std::invalid_argument("All rows in initializer list must have the same size.");
    }
    int j = 0;
    for (const auto &val : row) {
      v[i][j] = val;
      j++;
    }
    i++;
  }
}

template <class T>
Matrix<T>::Matrix(const char *filename) {
  if (!filename) {
    throw std::invalid_argument("Filename cannot be null.");
  }
  FILE *pFile = fopen(filename, "r");
  if (pFile == NULL) {
    throw std::runtime_error(std::string("File \"") + filename + "\" cannot be found.");
  }

  std::vector<char> line_buf(MAX_LINE_LENGTH);
  char *line = line_buf.data();
  int rows = 0;
  int columns = 0;

  while (fgets(line, MAX_LINE_LENGTH, pFile) != NULL) {
    int check = 0;
    char *token = strtok(line, " ,\t\r\n");
    while (token != NULL && token[0] >= 32) {
      check++;
      token = strtok(NULL, " ,\t\r\n");
    }
    if (check == 0) continue;
    if (rows == 0) {
      columns = check;
      rows = 1;
    } else if (check >= columns) {
      rows++;
    } else {
      break;
    }
  }
  fclose(pFile);

  if (rows < 1 || columns < 1) {
    throw std::runtime_error(std::string("File \"") + filename + "\" contains no valid matrix data.");
  }
  printf("Reading matrix from file \"%s\": %d rows, %d columns\n", filename, rows, columns);

  rows_ = rows;
  columns_ = columns;
  data_.resize(static_cast<size_t>(rows) * static_cast<size_t>(columns));
  syncRowPointers();

  pFile = fopen(filename, "r");
  if (pFile == NULL) {
    throw std::runtime_error("File cannot be found (second pass).");
  }
  int i = 0;
  while (i < rows && fgets(line, MAX_LINE_LENGTH, pFile) != NULL) {
    char *token = strtok(line, " ,\t\r\n");
    if (!token) continue;
    for (int j = 0; j < columns; j++) {
      if (j > 0) token = strtok(NULL, " ,\t\r\n");
      v[i][j] = token ? static_cast<T>(atof(token)) : static_cast<T>(0);
    }
    i++;
  }
  fclose(pFile);
}

template <class T>
Matrix<T>::Matrix(const std::string &filename) : Matrix(filename.c_str()) {}

template <class T>
Matrix<T> Matrix<T>::zeros(int rows, int columns) {
  return Matrix<T>(rows, columns, static_cast<T>(0));
}

template <class T>
Matrix<T> Matrix<T>::ones(int rows, int columns) {
  return Matrix<T>(rows, columns, static_cast<T>(1));
}

template <class T>
Matrix<T> Matrix<T>::eye(int n) {
  return Matrix<T>(n, n, "I");
}

template <class T>
Matrix<T> Matrix<T>::eye(int rows, int columns) {
  return Matrix<T>(rows, columns, "I");
}

template <class T>
Matrix<T> Matrix<T>::identity(int n) {
  return Matrix<T>(n, n, "I");
}

template <class T>
Matrix<T> Matrix<T>::rand(int rows, int columns) {
  return Matrix<T>(rows, columns, "rand");
}

template <class T>
Matrix<T> Matrix<T>::randn(int rows, int columns) {
  return Matrix<T>(rows, columns, "randn");
}

template <class T>
Matrix<T> Matrix<T>::randInt(int rows, int columns, int low, int high) {
  if (rows < 1 || columns < 1 || low > high) {
    throw std::invalid_argument("Invalid arguments to randInt().");
  }
  Matrix<T> M(rows, columns);
  std::uniform_int_distribution<int> dist(low, high);
  auto &rng = globalRng();
  for (size_t i = 0; i < M.data_.size(); ++i) {
    M.data_[i] = static_cast<T>(dist(rng));
  }
  return M;
}

template <class T>
Matrix<T> Matrix<T>::randperm(int rows, int columns) {
  return Matrix<T>(rows, columns, "randperm");
}

template <class T>
Matrix<T> Matrix<T>::diag(const Matrix<T> &vec_or_mat) {
  return vec_or_mat.diag();
}

template <class T>
Matrix<T> Matrix<T>::diag(const std::vector<T> &vec) {
  int n = static_cast<int>(vec.size());
  if (n < 1) throw std::invalid_argument("Vector for diag() cannot be empty.");
  Matrix<T> D(n, n, static_cast<T>(0));
  for (int i = 0; i < n; ++i) D.v[i][i] = vec[i];
  return D;
}

template <class T>
Matrix<T> Matrix<T>::linspace(T start, T end, int n) {
  if (n < 1) throw std::invalid_argument("linspace requires n >= 1.");
  Matrix<T> B(1, n);
  if (n == 1) {
    B.v[0][0] = end;
    return B;
  }
  double step = (static_cast<double>(end) - static_cast<double>(start)) / static_cast<double>(n - 1);
  for (int i = 0; i < n; ++i) {
    B.v[0][i] = static_cast<T>(static_cast<double>(start) + i * step);
  }
  B.v[0][n - 1] = end;
  return B;
}

template <class T>
Matrix<T> Matrix<T>::arange(T start, T stop, T step) {
  if (static_cast<double>(step) == 0.0) throw std::invalid_argument("arange step cannot be zero.");
  double d_start = static_cast<double>(start);
  double d_stop = static_cast<double>(stop);
  double d_step = static_cast<double>(step);
  if ((d_stop > d_start && d_step < 0.0) || (d_stop < d_start && d_step > 0.0)) {
    throw std::invalid_argument("arange step direction does not match start/stop.");
  }
  int n = static_cast<int>(std::ceil((d_stop - d_start) / d_step - 1e-12));
  if (n < 1) throw std::invalid_argument("arange resulted in empty range.");
  Matrix<T> B(1, n);
  for (int i = 0; i < n; ++i) {
    B.v[0][i] = static_cast<T>(d_start + i * d_step);
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::fromVector(int rows, int columns, const std::vector<T> &vec) {
  if (rows < 1 || columns < 1 || static_cast<int>(vec.size()) != rows * columns) {
    throw std::invalid_argument("Vector size does not match matrix dimensions.");
  }
  Matrix<T> B(rows, columns);
  B.data_ = vec;
  B.syncRowPointers();
  return B;
}

template <class T>
Matrix<T> Matrix<T>::fromArray(int rows, int columns, const T *arr) {
  if (!arr || rows < 1 || columns < 1) {
    throw std::invalid_argument("Invalid arguments to fromArray.");
  }
  Matrix<T> B(rows, columns);
  std::copy(arr, arr + rows * columns, B.data_.begin());
  return B;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {
  if (this != &other) {
    rows_ = other.rows_;
    columns_ = other.columns_;
    data_ = other.data_;
    syncRowPointers();
  }
  return *this;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other) noexcept {
  if (this != &other) {
    rows_ = other.rows_;
    columns_ = other.columns_;
    data_ = std::move(other.data_);
    syncRowPointers();
    other.rows_ = 0;
    other.columns_ = 0;
    other.v.clear();
  }
  return *this;
}

template <class T>
Matrix<T>::~Matrix() {}

template <class T>
void Matrix<T>::set(int r, int c, T value) {
  if (r < 0 || r >= rows_ || c < 0 || c >= columns_) {
    throw std::out_of_range("Invalid index in set(): r=" + std::to_string(r) + ", c=" + std::to_string(c));
  }
  v[r][c] = value;
}

template <class T>
T Matrix<T>::get(int r, int c) const {
  if (r < 0 || r >= rows_ || c < 0 || c >= columns_) {
    throw std::out_of_range("Matrix index out of range in get().");
  }
  return v[r][c];
}

template <class T>
T& Matrix<T>::operator()(int r, int c) {
  if (r < 0 || r >= rows_ || c < 0 || c >= columns_) {
    throw std::out_of_range("Matrix index out of range in operator()(r, c).");
  }
  return v[r][c];
}

template <class T>
const T& Matrix<T>::operator()(int r, int c) const {
  if (r < 0 || r >= rows_ || c < 0 || c >= columns_) {
    throw std::out_of_range("Matrix index out of range in operator()(r, c).");
  }
  return v[r][c];
}

template <class T>
T& Matrix<T>::operator()(int linear_idx) {
  if (linear_idx < 0 || linear_idx >= rows_ * columns_) {
    throw std::out_of_range("Matrix index out of range in operator()(idx).");
  }
  return data_[linear_idx];
}

template <class T>
const T& Matrix<T>::operator()(int linear_idx) const {
  if (linear_idx < 0 || linear_idx >= rows_ * columns_) {
    throw std::out_of_range("Matrix index out of range in operator()(idx).");
  }
  return data_[linear_idx];
}

template <class T>
int Matrix<T>::rows() const { return rows_; }

template <class T>
int Matrix<T>::columns() const { return columns_; }

template <class T>
bool Matrix<T>::isSymmetric(double tol) const {
  if (!isSquare()) return false;
  for (int i = 0; i < rows_; ++i) {
    for (int j = i + 1; j < columns_; ++j) {
      if (std::abs(static_cast<double>(v[i][j]) - static_cast<double>(v[j][i])) > tol) {
        return false;
      }
    }
  }
  return true;
}

template <class T>
bool Matrix<T>::equals(const Matrix<T> &other, double tol) const {
  if (rows_ != other.rows_ || columns_ != other.columns_) return false;
  for (size_t i = 0; i < data_.size(); ++i) {
    if (std::abs(static_cast<double>(data_[i]) - static_cast<double>(other.data_[i])) > tol) {
      return false;
    }
  }
  return true;
}

template <class T>
void Matrix<T>::fill(T value) {
  std::fill(data_.begin(), data_.end(), value);
}

template <class T>
void Matrix<T>::print() const {
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) {
      if (std::abs(static_cast<double>(v[i][j])) < EPSILON)
        printf("%10.4f ", 0.0);
      else
        printf("%10.4f ", static_cast<double>(v[i][j]));
    }
    printf("\n");
  }
}

template <class T>
Matrix<T> Matrix<T>::copy() const {
  return Matrix<T>(*this);
}

template <class T>
void Matrix<T>::saveTxt(const char *filename) const {
  if (!filename) throw std::invalid_argument("Filename cannot be null.");
  FILE *pFile = fopen(filename, "w");
  if (pFile == NULL) {
    throw std::runtime_error("File cannot be opened for writing.");
  }
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) {
      fprintf(pFile, "%f ", static_cast<double>(v[i][j]));
    }
    fprintf(pFile, "\n");
  }
  fclose(pFile);
  printf("Matrix saved to file \"%s\"\n", filename);
}

template <class T>
Matrix<T> Matrix<T>::transpose() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(columns_, rows_);
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) {
      B.v[j][i] = v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::reshape(int new_rows, int new_cols) const {
  if (new_rows < 1 || new_cols < 1 || new_rows * new_cols != rows_ * columns_) {
    throw std::invalid_argument("Invalid reshape dimensions.");
  }
  return Matrix<T>::fromVector(new_rows, new_cols, data_);
}

template <class T>
Matrix<T> Matrix<T>::flatten() const {
  if (empty()) return Matrix<T>();
  return reshape(rows_ * columns_, 1);
}

template <class T>
Matrix<T> Matrix<T>::sub(int r1, int r2, int c1, int c2) const {
  if (r1 < 0 || r2 >= rows_ || c1 < 0 || c2 >= columns_ || r1 > r2 || c1 > c2) {
    throw std::out_of_range("Matrix index out of range in sub().");
  }
  Matrix<T> B(r2 - r1 + 1, c2 - c1 + 1);
  for (int i = r1; i <= r2; i++) {
    for (int j = c1; j <= c2; j++) {
      B.v[i - r1][j - c1] = v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::row(int r) const {
  return sub(r, r, 0, columns_ - 1);
}

template <class T>
Matrix<T> Matrix<T>::col(int c) const {
  return sub(0, rows_ - 1, c, c);
}

template <class T>
void Matrix<T>::setRow(int r, const Matrix<T> &rowVec) {
  if (r < 0 || r >= rows_) throw std::out_of_range("Row index out of range in setRow().");
  if (rowVec.size() != columns_) throw std::invalid_argument("Dimension mismatch in setRow().");
  for (int j = 0; j < columns_; ++j) v[r][j] = rowVec.data_[j];
}

template <class T>
void Matrix<T>::setCol(int c, const Matrix<T> &colVec) {
  if (c < 0 || c >= columns_) throw std::out_of_range("Column index out of range in setCol().");
  if (colVec.size() != rows_) throw std::invalid_argument("Dimension mismatch in setCol().");
  for (int i = 0; i < rows_; ++i) v[i][c] = colVec.data_[i];
}

template <class T>
void Matrix<T>::swapRows(int r1, int r2) {
  if (r1 < 0 || r1 >= rows_ || r2 < 0 || r2 >= rows_) {
    throw std::out_of_range("Row index out of range in swapRows().");
  }
  if (r1 == r2) return;
  for (int j = 0; j < columns_; ++j) {
    std::swap(v[r1][j], v[r2][j]);
  }
}

template <class T>
void Matrix<T>::swapCols(int c1, int c2) {
  if (c1 < 0 || c1 >= columns_ || c2 < 0 || c2 >= columns_) {
    throw std::out_of_range("Column index out of range in swapCols().");
  }
  if (c1 == c2) return;
  for (int i = 0; i < rows_; ++i) {
    std::swap(v[i][c1], v[i][c2]);
  }
}

template <class T>
Matrix<T> Matrix<T>::diag() const {
  if (empty()) throw std::runtime_error("Cannot call diag() on empty matrix.");
  if ((rows_ == 1 && columns_ > 1) || (columns_ == 1 && rows_ > 1)) {
    int n = std::max(rows_, columns_);
    Matrix<T> D(n, n, static_cast<T>(0));
    for (int i = 0; i < n; ++i) D.v[i][i] = data_[i];
    return D;
  } else {
    int k = std::min(rows_, columns_);
    Matrix<T> d(k, 1);
    for (int i = 0; i < k; ++i) d.v[i][0] = v[i][i];
    return d;
  }
}

template <class T>
Matrix<T> Matrix<T>::triu(int k) const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_, static_cast<T>(0));
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < columns_; ++j) {
      if (j - i >= k) B.v[i][j] = v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::tril(int k) const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_, static_cast<T>(0));
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < columns_; ++j) {
      if (j - i <= k) B.v[i][j] = v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::abs() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::abs(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
T Matrix<T>::trace() const {
  T s = 0;
  for (int i = 0; i < std::min(rows_, columns_); i++)
    s += v[i][i];
  return s;
}

template <class T>
double Matrix<T>::fnorm() const {
  double s = 0;
  for (size_t i = 0; i < data_.size(); ++i) {
    double val = static_cast<double>(data_[i]);
    s += val * val;
  }
  return std::sqrt(s);
}

template <class T>
double Matrix<T>::pnorm(double p) const {
  if (p <= 0.0) throw std::invalid_argument("pnorm requires p > 0.");
  double s = 0;
  for (size_t i = 0; i < data_.size(); ++i) {
    s += std::pow(std::fabs(static_cast<double>(data_[i])), p);
  }
  return std::pow(s, 1.0 / p);
}

template <class T>
double Matrix<T>::norm1() const {
  double max_col = 0.0;
  for (int j = 0; j < columns_; ++j) {
    double col_sum = 0.0;
    for (int i = 0; i < rows_; ++i) col_sum += std::abs(static_cast<double>(v[i][j]));
    if (col_sum > max_col) max_col = col_sum;
  }
  return max_col;
}

template <class T>
double Matrix<T>::normInf() const {
  double max_row = 0.0;
  for (int i = 0; i < rows_; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < columns_; ++j) row_sum += std::abs(static_cast<double>(v[i][j]));
    if (row_sum > max_row) max_row = row_sum;
  }
  return max_row;
}

template <class T>
double Matrix<T>::norm2() const {
  if (empty()) return 0.0;
  Matrix<T> U, S, Vt;
  this->svd(U, S, Vt);
  return static_cast<double>(S.v[0][0]);
}

template <class T>
double Matrix<T>::cond() const {
  if (empty()) throw std::runtime_error("Cannot compute cond() of empty matrix.");
  Matrix<T> U, S, Vt;
  this->svd(U, S, Vt);
  int k = std::min(rows_, columns_);
  double s_max = static_cast<double>(S.v[0][0]);
  double s_min = static_cast<double>(S.v[k - 1][k - 1]);
  if (s_min <= 1e-15 * s_max) return INFINITY;
  return s_max / s_min;
}

template <class T>
T Matrix<T>::maxEl(int &r, int &c) const {
  if (empty()) throw std::runtime_error("Cannot call maxEl() on empty matrix.");
  T max_val = v[0][0];
  r = 0;
  c = 0;
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) {
      if (v[i][j] > max_val) {
        max_val = v[i][j];
        r = i;
        c = j;
      }
    }
  }
  return max_val;
}

template <class T>
T Matrix<T>::maxEl() const {
  int r, c;
  return maxEl(r, c);
}

template <class T>
T Matrix<T>::minEl(int &r, int &c) const {
  if (empty()) throw std::runtime_error("Cannot call minEl() on empty matrix.");
  T min_val = v[0][0];
  r = 0;
  c = 0;
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) {
      if (v[i][j] < min_val) {
        min_val = v[i][j];
        r = i;
        c = j;
      }
    }
  }
  return min_val;
}

template <class T>
T Matrix<T>::minEl() const {
  int r, c;
  return minEl(r, c);
}

template <class T>
double Matrix<T>::mean() const {
  if (empty()) throw std::runtime_error("Cannot call mean() on empty matrix.");
  return static_cast<double>(this->sum()) / static_cast<double>(rows_ * columns_);
}

template <class T>
T Matrix<T>::sum() const {
  T s = 0;
  for (size_t i = 0; i < data_.size(); ++i) {
    s += data_[i];
  }
  return s;
}

template <class T>
double Matrix<T>::var(bool sample) const {
  if (empty()) throw std::runtime_error("Cannot call var() on empty matrix.");
  int n = rows_ * columns_;
  if (sample && n <= 1) throw std::invalid_argument("Sample variance requires at least 2 elements.");
  double m = mean();
  double s = 0;
  for (size_t i = 0; i < data_.size(); ++i) {
    double diff = static_cast<double>(data_[i]) - m;
    s += diff * diff;
  }
  return s / static_cast<double>(sample ? (n - 1) : n);
}

template <class T>
double Matrix<T>::std(bool sample) const {
  return std::sqrt(var(sample));
}

template <class T>
Matrix<T> Matrix<T>::sumAxis(int axis) const {
  if (empty()) throw std::runtime_error("Cannot call sumAxis() on empty matrix.");
  if (axis == 0) {
    Matrix<T> S(1, columns_, static_cast<T>(0));
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < columns_; ++j) S.v[0][j] += v[i][j];
    }
    return S;
  } else if (axis == 1) {
    Matrix<T> S(rows_, 1, static_cast<T>(0));
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < columns_; ++j) S.v[i][0] += v[i][j];
    }
    return S;
  } else {
    throw std::invalid_argument("axis must be 0 (column-wise) or 1 (row-wise).");
  }
}

template <class T>
Matrix<double> Matrix<T>::meanAxis(int axis) const {
  if (empty()) throw std::runtime_error("Cannot call meanAxis() on empty matrix.");
  if (axis == 0) {
    Matrix<double> M(1, columns_, 0.0);
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < columns_; ++j) M.v[0][j] += static_cast<double>(v[i][j]);
    }
    M.divideNumberSelf(static_cast<double>(rows_));
    return M;
  } else if (axis == 1) {
    Matrix<double> M(rows_, 1, 0.0);
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < columns_; ++j) M.v[i][0] += static_cast<double>(v[i][j]);
    }
    M.divideNumberSelf(static_cast<double>(columns_));
    return M;
  } else {
    throw std::invalid_argument("axis must be 0 (column-wise) or 1 (row-wise).");
  }
}

template <class T>
Matrix<T> Matrix<T>::minAxis(int axis) const {
  if (empty()) throw std::runtime_error("Cannot call minAxis() on empty matrix.");
  if (axis == 0) {
    Matrix<T> M(1, columns_);
    for (int j = 0; j < columns_; ++j) {
      T mv = v[0][j];
      for (int i = 1; i < rows_; ++i) {
        if (v[i][j] < mv) mv = v[i][j];
      }
      M.v[0][j] = mv;
    }
    return M;
  } else if (axis == 1) {
    Matrix<T> M(rows_, 1);
    for (int i = 0; i < rows_; ++i) {
      T mv = v[i][0];
      for (int j = 1; j < columns_; ++j) {
        if (v[i][j] < mv) mv = v[i][j];
      }
      M.v[i][0] = mv;
    }
    return M;
  } else {
    throw std::invalid_argument("axis must be 0 (column-wise) or 1 (row-wise).");
  }
}

template <class T>
Matrix<T> Matrix<T>::maxAxis(int axis) const {
  if (empty()) throw std::runtime_error("Cannot call maxAxis() on empty matrix.");
  if (axis == 0) {
    Matrix<T> M(1, columns_);
    for (int j = 0; j < columns_; ++j) {
      T mv = v[0][j];
      for (int i = 1; i < rows_; ++i) {
        if (v[i][j] > mv) mv = v[i][j];
      }
      M.v[0][j] = mv;
    }
    return M;
  } else if (axis == 1) {
    Matrix<T> M(rows_, 1);
    for (int i = 0; i < rows_; ++i) {
      T mv = v[i][0];
      for (int j = 1; j < columns_; ++j) {
        if (v[i][j] > mv) mv = v[i][j];
      }
      M.v[i][0] = mv;
    }
    return M;
  } else {
    throw std::invalid_argument("axis must be 0 (column-wise) or 1 (row-wise).");
  }
}

template <class T>
Matrix<double> Matrix<T>::cov() const {
  if (empty()) throw std::runtime_error("Cannot call cov() on empty matrix.");
  int n = rows_;
  int p = columns_;
  Matrix<double> Xc = detail::toDoubleMatrix(*this);
  for (int j = 0; j < p; ++j) {
    double col_mean = 0.0;
    for (int i = 0; i < n; ++i) col_mean += Xc.v[i][j];
    col_mean /= static_cast<double>(n);
    for (int i = 0; i < n; ++i) Xc.v[i][j] -= col_mean;
  }
  Matrix<double> C = Xc.transpose() * Xc;
  double denom = (n > 1) ? static_cast<double>(n - 1) : 1.0;
  C.divideNumberSelf(denom);
  return C;
}

template <class T>
Matrix<double> Matrix<T>::corrcoef() const {
  Matrix<double> C = this->cov();
  int p = C.rows();
  Matrix<double> R(p, p, 0.0);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      double denom = std::sqrt(C.v[i][i] * C.v[j][j]);
      R.v[i][j] = (denom > 1e-15) ? (C.v[i][j] / denom) : ((i == j) ? 1.0 : 0.0);
    }
  }
  return R;
}

template <class T>
void Matrix<T>::addNumberSelf(T value) {
  for (size_t i = 0; i < data_.size(); ++i) data_[i] += value;
}

template <class T>
void Matrix<T>::subtractNumberSelf(T value) {
  for (size_t i = 0; i < data_.size(); ++i) data_[i] -= value;
}

template <class T>
void Matrix<T>::multiplyNumberSelf(T value) {
  for (size_t i = 0; i < data_.size(); ++i) data_[i] *= value;
}

template <class T>
void Matrix<T>::divideNumberSelf(T value) {
  if (static_cast<double>(value) == 0.0) {
    throw std::invalid_argument("Division by zero in divideNumberSelf().");
  }
  for (size_t i = 0; i < data_.size(); ++i) data_[i] /= value;
}

template <class T>
Matrix<T> Matrix<T>::exp() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::exp(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::log() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::log(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::sqrt() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::sqrt(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::power(double p) const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::pow(static_cast<double>(data_[i]), p));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::sin() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::sin(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::cos() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::cos(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::tanh() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::tanh(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::apply(std::function<T(T)> fn) const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = fn(data_[i]);
  }
  return B;
}

template <class T>
void Matrix<T>::addMatrixSelf(const Matrix<T> &A) {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Unmatched matrix sizes in matrix summation.");
  }
  for (size_t i = 0; i < data_.size(); ++i) data_[i] += A.data_[i];
}

template <class T>
void Matrix<T>::subtractMatrixSelf(const Matrix<T> &A) {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Unmatched matrix sizes in matrix subtraction.");
  }
  for (size_t i = 0; i < data_.size(); ++i) data_[i] -= A.data_[i];
}

template <class T>
void Matrix<T>::dotMultiplyMatrixSelf(const Matrix<T> &A) {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Unmatched matrix sizes in matrix dot multiplication.");
  }
  for (size_t i = 0; i < data_.size(); ++i) data_[i] *= A.data_[i];
}

template <class T>
void Matrix<T>::dotDivideMatrixSelf(const Matrix<T> &A) {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Unmatched matrix sizes in matrix dot division.");
  }
  for (size_t i = 0; i < data_.size(); ++i) data_[i] /= A.data_[i];
}

template <class T>
Matrix<T> Matrix<T>::addMatrixNew(const Matrix<T> &A) const {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Matrix dimensions must match for addition.");
  }
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) B.data_[i] = data_[i] + A.data_[i];
  return B;
}

template <class T>
Matrix<T> Matrix<T>::subtractMatrixNew(const Matrix<T> &A) const {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Matrix dimensions must match for subtraction.");
  }
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) B.data_[i] = data_[i] - A.data_[i];
  return B;
}

template <class T>
Matrix<T> Matrix<T>::dotMultiplyMatrixNew(const Matrix<T> &A) const {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Matrix dimensions must match for dot multiplication.");
  }
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) B.data_[i] = data_[i] * A.data_[i];
  return B;
}

template <class T>
Matrix<T> Matrix<T>::dotDivideMatrixNew(const Matrix<T> &A) const {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Matrix dimensions must match for dot division.");
  }
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) B.data_[i] = data_[i] / A.data_[i];
  return B;
}

/* Cache-friendly i-k-j matrix multiplication */
template <class T>
Matrix<T> Matrix<T>::multiplyMatrixNew(const Matrix<T> &A) const {
  if (columns_ != A.rows()) {
    throw std::invalid_argument("Matrix dimensions must match for multiplication.");
  }
  int a_cols = A.columns();
  Matrix<T> B(rows_, a_cols, static_cast<T>(0));
  for (int i = 0; i < rows_; i++) {
    T *b_row = B.v[i];
    const T *this_row = v[i];
    for (int k = 0; k < columns_; k++) {
      T aik = this_row[k];
      const T *a_row = A.v[k];
      for (int j = 0; j < a_cols; j++) {
        b_row[j] += aik * a_row[j];
      }
    }
  }
  return B;
}

template <class T>
T Matrix<T>::dot(const Matrix<T> &other) const {
  if (size() != other.size()) {
    throw std::invalid_argument("Sizes must match for dot().");
  }
  T s = 0;
  for (size_t i = 0; i < data_.size(); ++i) s += data_[i] * other.data_[i];
  return s;
}

template <class T>
Matrix<T> Matrix<T>::outer(const Matrix<T> &other) const {
  if (empty() || other.empty()) throw std::invalid_argument("Cannot compute outer() of empty matrix.");
  int m = size();
  int n = other.size();
  Matrix<T> C(m, n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C.v[i][j] = data_[i] * other.data_[j];
    }
  }
  return C;
}

template <class T>
Matrix<T> Matrix<T>::cross(const Matrix<T> &other) const {
  if (size() != 3 || other.size() != 3) {
    throw std::invalid_argument("cross() requires 3-element vectors.");
  }
  Matrix<T> c(rows_, columns_);
  c.data_[0] = data_[1] * other.data_[2] - data_[2] * other.data_[1];
  c.data_[1] = data_[2] * other.data_[0] - data_[0] * other.data_[2];
  c.data_[2] = data_[0] * other.data_[1] - data_[1] * other.data_[0];
  return c;
}

template <class T>
Matrix<T> Matrix<T>::kron(const Matrix<T> &other) const {
  if (empty() || other.empty()) throw std::invalid_argument("Cannot compute kron() of empty matrix.");
  Matrix<T> K(rows_ * other.rows_, columns_ * other.columns_);
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < columns_; ++j) {
      T a_ij = v[i][j];
      for (int r = 0; r < other.rows_; ++r) {
        for (int c = 0; c < other.columns_; ++c) {
          K.v[i * other.rows_ + r][j * other.columns_ + c] = a_ij * other.v[r][c];
        }
      }
    }
  }
  return K;
}

template <class T>
Matrix<T> Matrix<T>::concatenateRight(const Matrix<T> &A) const {
  if (rows_ != A.rows()) {
    throw std::invalid_argument("Matrix rows must match for concatenateRight().");
  }
  Matrix<T> B(rows_, columns_ + A.columns());
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) B.v[i][j] = v[i][j];
    for (int j = 0; j < A.columns(); j++) B.v[i][j + columns_] = A.v[i][j];
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::concatenateBottom(const Matrix<T> &A) const {
  if (columns_ != A.columns()) {
    throw std::invalid_argument("Matrix columns must match for concatenateBottom().");
  }
  Matrix<T> B(rows_ + A.rows(), columns_);
  for (int j = 0; j < columns_; j++) {
    for (int i = 0; i < rows_; i++) B.v[i][j] = v[i][j];
    for (int i = 0; i < A.rows(); i++) B.v[i + rows_][j] = A.v[i][j];
  }
  return B;
}

template <class T>
void Matrix<T>::lu(Matrix<T> &L, Matrix<T> &U, Matrix<T> &P) const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("LU decomposition requires square matrix.");
  }
  int n = rows_;
  Matrix<double> LU_d;
  std::vector<int> piv;
  detail::luDecomposeDouble(detail::toDoubleMatrix(*this), LU_d, piv);

  L = Matrix<T>(n, n, static_cast<T>(0));
  U = Matrix<T>(n, n, static_cast<T>(0));
  P = Matrix<T>(n, n, static_cast<T>(0));
  for (int i = 0; i < n; ++i) {
    P.v[i][piv[i]] = static_cast<T>(1);
    L.v[i][i] = static_cast<T>(1);
    for (int j = 0; j < i; ++j) L.v[i][j] = static_cast<T>(LU_d.v[i][j]);
    for (int j = i; j < n; ++j) U.v[i][j] = static_cast<T>(LU_d.v[i][j]);
  }
}

template <class T>
Matrix<T> Matrix<T>::cholesky() const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("Cholesky decomposition requires a square matrix.");
  }
  if (!isSymmetric(1e-7)) {
    throw std::invalid_argument("Cholesky decomposition requires a symmetric matrix.");
  }
  int n = rows_;
  Matrix<T> L(n, n, static_cast<T>(0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = static_cast<double>(v[i][j]);
      for (int k = 0; k < j; ++k) {
        sum -= static_cast<double>(L.v[i][k]) * static_cast<double>(L.v[j][k]);
      }
      if (i == j) {
        if (sum <= 1e-15) {
          throw std::runtime_error("Matrix is not positive definite in cholesky().");
        }
        L.v[i][i] = static_cast<T>(std::sqrt(sum));
      } else {
        L.v[i][j] = static_cast<T>(sum / static_cast<double>(L.v[j][j]));
      }
    }
  }
  return L;
}

template <class T>
void Matrix<T>::cholesky(Matrix<T> &L) const {
  L = this->cholesky();
}

template <class T>
T Matrix<T>::determinant() const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("Determinant requires square matrix.");
  }
  int n = rows_;
  Matrix<double> LU_d;
  std::vector<int> piv;
  int swaps = detail::luDecomposeDouble(detail::toDoubleMatrix(*this), LU_d, piv);
  double det = (swaps % 2 == 0) ? 1.0 : -1.0;
  for (int i = 0; i < n; ++i) {
    det *= LU_d.v[i][i];
  }
  return static_cast<T>(det);
}

template <class T>
Matrix<T> Matrix<T>::solve(const Matrix<T> &B) const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("Matrix must be square to solve system.");
  }
  if (rows_ != B.rows()) {
    throw std::invalid_argument("Matrix row dimensions must agree.");
  }
  int n = rows_;
  int rhs = B.columns();
  Matrix<double> LU_d;
  std::vector<int> piv;
  detail::luDecomposeDouble(detail::toDoubleMatrix(*this), LU_d, piv);

  for (int i = 0; i < n; ++i) {
    if (std::fabs(LU_d.v[i][i]) <= 1e-14) {
      throw std::runtime_error("Matrix is singular.");
    }
  }

  Matrix<double> B_d = detail::toDoubleMatrix(B);
  Matrix<double> X_d(n, rhs, 0.0);

  for (int col = 0; col < rhs; ++col) {
    for (int i = 0; i < n; ++i) {
      double sum = B_d.v[piv[i]][col];
      for (int j = 0; j < i; ++j) sum -= LU_d.v[i][j] * X_d.v[j][col];
      X_d.v[i][col] = sum;
    }
    for (int i = n - 1; i >= 0; --i) {
      double sum = X_d.v[i][col];
      for (int j = i + 1; j < n; ++j) sum -= LU_d.v[i][j] * X_d.v[j][col];
      X_d.v[i][col] = sum / LU_d.v[i][i];
    }
  }
  return detail::fromDoubleMatrix<T>(X_d);
}

template <class T>
Matrix<T> Matrix<T>::inverse() const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("Inverse requires square matrix.");
  }
  return this->solve(Matrix<T>::eye(rows_));
}

template <class T>
Matrix<T> Matrix<T>::pinv(double tol) const {
  if (empty()) throw std::invalid_argument("Cannot compute pinv() of empty matrix.");
  Matrix<T> U, S, Vt;
  this->svd(U, S, Vt);
  int k = S.rows();
  double sigma_max = (k > 0) ? static_cast<double>(S.v[0][0]) : 0.0;
  double cutoff = (tol >= 0.0) ? tol : (static_cast<double>(std::max(rows_, columns_)) * std::max(sigma_max, 1.0) * 1e-12);
  Matrix<T> V = Vt.transpose();
  Matrix<T> Ut = U.transpose();
  for (int i = 0; i < k; ++i) {
    double s = static_cast<double>(S.v[i][i]);
    double s_inv = (s > cutoff) ? (1.0 / s) : 0.0;
    for (int j = 0; j < V.rows(); ++j) {
      V.v[j][i] = static_cast<T>(static_cast<double>(V.v[j][i]) * s_inv);
    }
  }
  return V * Ut;
}

template <class T>
Matrix<T> Matrix<T>::lstsq(const Matrix<T> &B) const {
  if (empty() || B.empty() || rows_ != B.rows()) {
    throw std::invalid_argument("Matrix row dimensions must agree in lstsq().");
  }
  return this->pinv() * B;
}

template <class T>
void Matrix<T>::qr(Matrix<T> &Q, Matrix<T> &R) const {
  if (empty()) {
    throw std::invalid_argument("Cannot compute QR of empty matrix.");
  }
  Matrix<double> Q_d, R_d;
  detail::qrDouble(detail::toDoubleMatrix(*this), Q_d, R_d);
  Q = detail::fromDoubleMatrix<T>(Q_d);
  R = detail::fromDoubleMatrix<T>(R_d);
}

template <class T>
void Matrix<T>::eigen(Matrix<T> &V, Matrix<T> &D) const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("Eigen decomposition requires square matrix.");
  }
  if (!isSymmetric(1e-7)) {
    throw std::invalid_argument("Eigen decomposition requires a symmetric matrix.");
  }
  Matrix<double> V_d, D_d;
  detail::eigenSymmetricDouble(detail::toDoubleMatrix(*this), V_d, D_d);
  V = detail::fromDoubleMatrix<T>(V_d);
  D = detail::fromDoubleMatrix<T>(D_d);
}

template <class T>
Matrix<T> Matrix<T>::eigenvalues() const {
  Matrix<T> V, D;
  this->eigen(V, D);
  Matrix<T> eigVals(rows_, 1);
  for (int i = 0; i < rows_; i++) {
    eigVals.v[i][0] = D.v[i][i];
  }
  return eigVals;
}

template <class T>
void Matrix<T>::svd(Matrix<T> &U, Matrix<T> &S, Matrix<T> &Vt) const {
  if (empty()) {
    throw std::invalid_argument("Cannot compute SVD of empty matrix.");
  }
  Matrix<double> U_d, S_d, Vt_d;
  detail::svdDouble(detail::toDoubleMatrix(*this), U_d, S_d, Vt_d);
  U = detail::fromDoubleMatrix<T>(U_d);
  S = detail::fromDoubleMatrix<T>(S_d);
  Vt = detail::fromDoubleMatrix<T>(Vt_d);
}

template <class T>
int Matrix<T>::rank(double tol) const {
  if (empty()) return 0;
  Matrix<T> U, S, Vt;
  this->svd(U, S, Vt);
  int k = std::min(rows_, columns_);
  double sigma_max = (k > 0) ? static_cast<double>(S.v[0][0]) : 0.0;
  double cutoff = (tol >= 0.0) ? tol : (static_cast<double>(std::max(rows_, columns_)) * std::max(sigma_max, 1.0) * 1e-12);
  int r = 0;
  for (int i = 0; i < k; i++) {
    if (std::abs(static_cast<double>(S.v[i][i])) > cutoff) r++;
  }
  return r;
}

template <class T>
Matrix<T> Matrix<T>::orth(double tol) const {
  if (empty()) return Matrix<T>();
  Matrix<T> U, S, Vt;
  this->svd(U, S, Vt);
  int r = this->rank(tol);
  if (r == 0) return Matrix<T>();
  return U.sub(0, rows_ - 1, 0, r - 1);
}

template <class T>
Matrix<T> Matrix<T>::nullSpace(double tol) const {
  if (empty()) return Matrix<T>();
  Matrix<double> A_d = detail::toDoubleMatrix(*this);
  int m = rows_;
  int n = columns_;

  // Full right singular vectors from A^T * A (n x n symmetric)
  Matrix<double> AtA = A_d.transpose() * A_d;
  Matrix<double> V_d, D_d;
  detail::eigenSymmetricDouble(AtA, V_d, D_d);

  double sigma_max = (D_d.v[0][0] > 0.0) ? std::sqrt(D_d.v[0][0]) : 0.0;
  double cutoff = (tol >= 0.0) ? tol : (static_cast<double>(std::max(m, n)) * std::max(sigma_max, 1.0) * 1e-10);
  int r = 0;
  for (int i = 0; i < n; ++i) {
    double s = (D_d.v[i][i] > 0.0) ? std::sqrt(D_d.v[i][i]) : 0.0;
    if (s > cutoff) r++;
  }
  int nullity = n - r;
  if (nullity <= 0) return Matrix<T>();
  return detail::fromDoubleMatrix<T>(V_d.sub(0, n - 1, r, n - 1));
}

template <class T>
Matrix<T> Matrix<T>::expm() const {
  if (!isSquare()) {
    throw std::invalid_argument("Matrix exponential requires a square matrix.");
  }
  int n = rows_;
  Matrix<double> A_d = detail::toDoubleMatrix(*this);
  double n_inf = A_d.normInf();
  int s = 0;
  if (n_inf > 0.5) {
    s = static_cast<int>(std::ceil(std::log2(n_inf / 0.5)));
    if (s < 0) s = 0;
  }
  double scale = std::ldexp(1.0, -s);
  Matrix<double> As = A_d * scale;
  Matrix<double> As2 = As * As;
  Matrix<double> As4 = As2 * As2;
  Matrix<double> As6 = As4 * As2;
  Matrix<double> I = Matrix<double>::eye(n);

  static const double b[7] = {
    1.0,
    1.0 / 2.0,
    3.0 / 26.0,
    1.0 / 78.0,
    1.0 / 1144.0,
    1.0 / 34320.0,
    1.0 / 2059200.0
  };

  Matrix<double> U(n, n, 0.0);
  Matrix<double> V(n, n, 0.0);
  for (int idx = 0; idx < n * n; ++idx) {
    double i_val = I.data()[idx];
    double a2 = As2.data()[idx];
    double a4 = As4.data()[idx];
    double a6 = As6.data()[idx];
    U.data()[idx] = b[1] * i_val + b[3] * a2 + b[5] * a4;
    V.data()[idx] = b[0] * i_val + b[2] * a2 + b[4] * a4 + b[6] * a6;
  }
  Matrix<double> U_full = As * U;
  Matrix<double> P = V + U_full;
  Matrix<double> Q = V - U_full;
  Matrix<double> E = Q.solve(P);

  for (int k = 0; k < s; ++k) {
    E = E * E;
  }
  return detail::fromDoubleMatrix<T>(E);
}

template <class T>
void Matrix<T>::pca(int n_components, Matrix<T> &projected, Matrix<T> &components, Matrix<T> &explained_variance) const {
  if (empty() || n_components < 1 || n_components > columns_) {
    throw std::invalid_argument("Invalid n_components for pca().");
  }
  int n = rows_;
  int p = columns_;
  Matrix<double> Xc = detail::toDoubleMatrix(*this);
  for (int j = 0; j < p; ++j) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) m += Xc.v[i][j];
    m /= static_cast<double>(n);
    for (int i = 0; i < n; ++i) Xc.v[i][j] -= m;
  }

  Matrix<double> C = Xc.cov();
  Matrix<double> V_d, D_d;
  detail::eigenSymmetricDouble(C, V_d, D_d);

  Matrix<double> comp_d = V_d.sub(0, p - 1, 0, n_components - 1);
  Matrix<double> proj_d = Xc * comp_d;
  Matrix<double> var_d(n_components, 1);
  for (int k = 0; k < n_components; ++k) {
    var_d.v[k][0] = (D_d.v[k][k] > 0.0) ? D_d.v[k][k] : 0.0;
  }

  projected = detail::fromDoubleMatrix<T>(proj_d);
  components = detail::fromDoubleMatrix<T>(comp_d);
  explained_variance = detail::fromDoubleMatrix<T>(var_d);
}

/* Operator overloads */

template <class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const {
  return this->addMatrixNew(other);
}

template <class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) const {
  return this->subtractMatrixNew(other);
}

template <class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const {
  return this->multiplyMatrixNew(other);
}

template <class T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T> &other) {
  this->addMatrixSelf(other);
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T> &other) {
  this->subtractMatrixSelf(other);
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T> &other) {
  *this = this->multiplyMatrixNew(other);
  return *this;
}

template <class T>
Matrix<T> Matrix<T>::operator-() const {
  Matrix<T> B = this->copy();
  B.multiplyNumberSelf(static_cast<T>(-1));
  return B;
}

template <class T>
Matrix<T> Matrix<T>::operator+(T scalar) const {
  Matrix<T> B = this->copy();
  B.addNumberSelf(scalar);
  return B;
}

template <class T>
Matrix<T> Matrix<T>::operator-(T scalar) const {
  Matrix<T> B = this->copy();
  B.subtractNumberSelf(scalar);
  return B;
}

template <class T>
Matrix<T> Matrix<T>::operator*(T scalar) const {
  Matrix<T> B = this->copy();
  B.multiplyNumberSelf(scalar);
  return B;
}

template <class T>
Matrix<T> Matrix<T>::operator/(T scalar) const {
  Matrix<T> B = this->copy();
  B.divideNumberSelf(scalar);
  return B;
}

template <class T>
Matrix<T>& Matrix<T>::operator+=(T scalar) {
  this->addNumberSelf(scalar);
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator-=(T scalar) {
  this->subtractNumberSelf(scalar);
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator*=(T scalar) {
  this->multiplyNumberSelf(scalar);
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator/=(T scalar) {
  this->divideNumberSelf(scalar);
  return *this;
}

template <class T>
bool Matrix<T>::operator==(const Matrix<T> &other) const {
  return this->equals(other, 1e-9);
}

template <class T>
bool Matrix<T>::operator!=(const Matrix<T> &other) const {
  return !this->equals(other, 1e-9);
}

/* Non-member scalar-matrix operators and stream output */

template <class T>
inline Matrix<T> operator+(T scalar, const Matrix<T> &M) {
  return M + scalar;
}

template <class T>
inline Matrix<T> operator-(T scalar, const Matrix<T> &M) {
  return (-M) + scalar;
}

template <class T>
inline Matrix<T> operator*(T scalar, const Matrix<T> &M) {
  return M * scalar;
}

template <class T>
inline std::ostream& operator<<(std::ostream &os, const Matrix<T> &M) {
  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j < M.columns(); ++j) {
      double val = static_cast<double>(M.get(i, j));
      if (std::abs(val) < EPSILON) val = 0.0;
      os << std::setw(10) << std::fixed << std::setprecision(4) << val << " ";
    }
    os << "\n";
  }
  return os;
}

/**********************************************
 * Algorithm part
 **********************************************/

// Calculate the pairwise interpoint Euclidean distances
// X is data matrix, D is distance matrix
inline void EuclideanDistanceMatrix(const Matrix<double> &X, Matrix<double> &D) {
  if (X.rows() != D.rows() || X.rows() != D.columns()) {
    throw std::invalid_argument("Invalid distance matrix dimension.");
  }

  for (int i = 0; i < D.rows(); i++)
    D.v[i][i] = 0.0;

  for (int i = 0; i < D.rows() - 1; i++) {
    for (int j = i + 1; j < D.columns(); j++) {
      double temp = 0.0;
      for (int k = 0; k < X.columns(); k++) {
        double diff = X.v[i][k] - X.v[j][k];
        temp += diff * diff;
      }
      D.v[i][j] = std::sqrt(temp);
    }
  }

  for (int i = 1; i < D.rows(); i++) {
    for (int j = 0; j < i; j++) {
      D.v[i][j] = D.v[j][i];
    }
  }
}

// Copy all elements of X to Y
inline void ElementCopy(const Matrix<double> &X, Matrix<double> &Y) {
  if (X.rows() != Y.rows() || X.columns() != Y.columns()) {
    throw std::invalid_argument("Invalid matrix dimension.");
  }
  for (int i = 0; i < X.rows(); i++) {
    for (int j = 0; j < X.columns(); j++) {
      Y.v[i][j] = X.v[i][j];
    }
  }
}

// Classical (Torgerson) Metric Multidimensional Scaling
inline Matrix<double> MDS_Classical(const Matrix<double> &D, int dim) {
  if (D.rows() != D.columns()) {
    throw std::invalid_argument("Input distance matrix to MDS is not square.");
  }
  if (dim < 1 || dim > D.rows()) {
    throw std::invalid_argument("Invalid dimension for MDS.");
  }
  int n = D.rows();
  Matrix<double> D2(n, n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      D2.v[i][j] = D.v[i][j] * D.v[i][j];
    }
  }

  std::vector<double> row_means(n, 0.0);
  double grand_mean = 0.0;
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < n; ++j) s += D2.v[i][j];
    row_means[i] = s / static_cast<double>(n);
    grand_mean += s;
  }
  grand_mean /= static_cast<double>(n * n);

  Matrix<double> B(n, n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      B.v[i][j] = -0.5 * (D2.v[i][j] - row_means[i] - row_means[j] + grand_mean);
    }
  }

  Matrix<double> V, EigD;
  detail::eigenSymmetricDouble(B, V, EigD);

  Matrix<double> X(n, dim, 0.0);
  for (int k = 0; k < dim; ++k) {
    double lam = EigD.v[k][k];
    double scale = (lam > 0.0) ? std::sqrt(lam) : 0.0;
    for (int i = 0; i < n; ++i) {
      X.v[i][k] = V.v[i][k] * scale;
    }
  }
  return X;
}

// Multidimensional scaling (MDS)
// This function re-implements Laurens van der Maaten's MDS in his Matlab Toolbox for Dimensionality Reduction
inline Matrix<double> MDS_UCF(const Matrix<double> &D, const Matrix<double> *X0, int dim, int iter) {
  if (D.rows() != D.columns()) {
    throw std::invalid_argument("Input distance matrix to MDS is not square.");
  }
  if (dim < 1) {
    throw std::invalid_argument("Invalid dimension for MDS.");
  }
  if (iter < 1) {
    throw std::invalid_argument("Invalid number of iterations for MDS.");
  }

  Matrix<double> X;
  if (X0 != NULL) {
    if (X0->rows() != D.rows() || X0->columns() != dim) {
      throw std::invalid_argument("Input initialization to MDS has invalid dimension.");
    }
    X = X0->copy();
  } else {
    X = Matrix<double>(D.rows(), dim, "rand");
    double D_mean = D.mean();
    X.addNumberSelf(-0.5);
    X.multiplyNumberSelf(0.1 * D_mean / (1.0 / 3.0 * std::sqrt(static_cast<double>(dim))));
  }

  double lr = 0.05; // learning rate
  double r = 2.0;   // metric
  int n = D.rows(); // number of vectors

  Matrix<double> dh(n, n, 0.0);
  Matrix<double> pmat(n, dim);
  Matrix<double> dhdum(n, 1);
  Matrix<double> dhmat(n > 1 ? n - 1 : 1, dim, 0.0);

  Matrix<int> RP(n, iter, "randperm");
  int i, j;
  double temp;
  int m;

  printf("\n[MDS] Iterations:\n");
  for (int it = 0; it < iter; it++) {
    if (it % 10 == 0)
      printf("\n[MDS] ");
    printf("%3d  ", it + 1);
    for (int rp = 0; rp < n; rp++) {
      m = RP.v[rp][it] - 1;

      for (i = 0; i < n; i++) {
        for (j = 0; j < dim; j++) {
          pmat.v[i][j] = X.v[m][j] - X.v[i][j];
        }
      }

      for (i = 0; i < n; i++) {
        temp = 0;
        for (j = 0; j < dim; j++) {
          temp += std::pow(std::fabs(pmat.v[i][j]), r);
        }
        dhdum.v[i][0] = std::pow(temp, 1.0 / r);
      }

      for (i = 0; i < n; i++) {
        if (i == m) continue;
        dh.v[m][i] = dhdum.v[i][0];
        dh.v[i][m] = dhdum.v[i][0];
      }

      for (i = 0; i < n - 1; i++) {
        int ii = (i >= m) ? (i + 1) : i;
        double dist = dhdum.v[ii][0];
        temp = (dist > 1e-12) ? (lr * (dist - D.v[ii][m]) * std::pow(dist, 1.0 - r)) : 0.0;
        for (j = 0; j < dim; j++) {
          dhmat.v[i][j] = temp;
        }
      }

      for (i = 0; i < n - 1; i++) {
        int ii = (i >= m) ? (i + 1) : i;
        for (j = 0; j < dim; j++) {
          temp = X.v[ii][j];
          temp += dhmat.v[i][j] * std::pow(std::fabs(pmat.v[ii][j]), r - 1.0) * sign<double>(pmat.v[ii][j]);
          X.v[ii][j] = temp;
        }
      }
    }
  }

  printf("\n");
  return X;
}

// Multidimensional scaling (MDS) with SMACOF
// This code re-implements Michael Bronstein's SMACOF in his Matlab Toolbox for Surface Comparison and Analysis
inline Matrix<double> MDS_SMACOF(const Matrix<double> &D, const Matrix<double> *X0, int dim, int iter) {
  if (D.rows() != D.columns()) {
    throw std::invalid_argument("Input distance matrix to MDS is not square.");
  }
  if (dim < 1) {
    throw std::invalid_argument("Invalid dimension for MDS.");
  }
  if (iter < 1) {
    throw std::invalid_argument("Invalid number of iterations for MDS.");
  }

  Matrix<double> X;
  if (X0 != NULL) {
    if (X0->rows() != D.rows() || X0->columns() != dim) {
      throw std::invalid_argument("Input initialization to MDS has invalid dimension.");
    }
    X = X0->copy();
  } else {
    X = Matrix<double>(D.rows(), dim, "rand");
    double D_mean = D.mean();
    X.addNumberSelf(-0.5);
    X.multiplyNumberSelf(0.1 * D_mean / (1.0 / 3.0 * std::sqrt(static_cast<double>(dim))));
  }

  Matrix<double> Z = X.copy();
  Matrix<double> D_(D.rows(), D.columns());
  Matrix<double> B(D.rows(), D.columns());
  int i, j, k;
  double temp;

  EuclideanDistanceMatrix(X, D_);

  printf("\n[MDS] Iterations:\n");
  for (int it = 0; it < iter; it++) {
    if (it % 10 == 0)
      printf("\n[MDS] ");
    printf("%3d  ", it + 1);

    for (i = 0; i < D.rows(); i++) {
      for (j = 0; j < D.columns(); j++) {
        if (i == j || std::fabs(D_.v[i][j]) < EPSILON) {
          B.v[i][j] = 0.0;
        } else {
          B.v[i][j] = -D.v[i][j] / D_.v[i][j];
        }
      }
    }

    for (j = 0; j < D.columns(); j++) {
      temp = 0;
      for (i = 0; i < D.rows(); i++) {
        temp += B.v[i][j];
      }
      B.v[j][j] = -temp;
    }

    for (i = 0; i < X.rows(); i++) {
      for (j = 0; j < X.columns(); j++) {
        temp = 0;
        for (k = 0; k < B.columns(); k++) {
          temp += (B.v[i][k] * Z.v[k][j]);
        }
        X.v[i][j] = temp / static_cast<double>(D.rows());
      }
    }

    EuclideanDistanceMatrix(X, D_);
    Z = X;
  }

  printf("\n");
  return X;
}

} // namespace smat

#endif /* SIMPLE_MATRIX_H */
