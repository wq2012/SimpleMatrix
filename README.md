# SimpleMatrix [![C/C++ CI](https://github.com/wq2012/SimpleMatrix/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/wq2012/SimpleMatrix/actions/workflows/c-cpp.yml)

**SimpleMatrix** is a lightweight, zero-dependency, single-header matrix and linear algebra library for **both Pure C (C99/C11) and C++ (C++11/14/17/20/23)**. Drop `SimpleMatrix.h` into any C or C++ project and immediately access a complete suite of matrix operations, decompositions, linear solvers, matrix exponentials, statistics, Principal Component Analysis (PCA), and Multidimensional Scaling (MDS).

---

## Table of Contents
* [Key Features](#key-features)
* [Integration](#integration)
* [Quick Start](#quick-start)
  * [C++ Quick Start](#c-quick-start)
  * [Pure C Quick Start](#pure-c-quick-start)
* [C++ API Reference (`smat::Matrix<T>`)](#c-api-reference-smatmatrixt)
  * [Constructors & Factory Methods](#constructors--factory-methods)
  * [Element Access, Indexing & Properties](#element-access-indexing--properties)
  * [Shape, Slicing & Concatenation](#shape-slicing--concatenation)
  * [Operators, Arithmetic & Math Functions](#operators-arithmetic--math-functions)
  * [Vector & Matrix Products](#vector--matrix-products)
  * [Norms, Reductions & Statistics](#norms-reductions--statistics)
  * [Linear Algebra Decompositions & Solvers](#linear-algebra-decompositions--solvers)
  * [Dimensionality Reduction (PCA & MDS)](#dimensionality-reduction-pca--mds)
* [Pure C API Reference (`smat_t`)](#pure-c-api-reference-smat_t)
  * [Types & Error Handling](#types--error-handling)
  * [Creation & Memory Management](#creation--memory-management)
  * [Accessors, Slicing & File I/O](#accessors-slicing--file-io)
  * [Arithmetic, Products & Statistics](#arithmetic-products--statistics)
  * [Linear Algebra, PCA & MDS in C](#linear-algebra-pca--mds-in-c)
* [Examples & Demos](#examples--demos)
* [Building and Testing](#building-and-testing)
* [Citation](#citation)
* [Copyright](#copyright)

---

## Key Features
* **Single-Header Dual C / C++ Support:** Include `SimpleMatrix.h` directly in `.c` files (compiled with `gcc`/`clang`) or `.cpp` files (compiled with `g++`/`clang++`).
* **Zero External Dependencies:** Uses only the standard C/C++ library.
* **High Efficiency & Cache Locality:** Contiguous row-major storage with precomputed row pointers (`v[r][c]`) and cache-friendly `i-k-j` matrix multiplication.
* **Memory Safe:** Automatic RAII management (`std::vector` backing with move semantics) in C++, and clean `smat_create` / `smat_free` lifecycle with null/bounds safety in Pure C.
* **Comprehensive Linear Algebra:**
  * **Decompositions:** LU ($PA = LU$), Cholesky ($A = LL^T$), QR ($A = QR$ for any $m \times n$), Symmetric Eigenvalue ($AV = VD$ with sorted eigenvalues), and Singular Value Decomposition ($A = USV^T$ with sorted singular values and full orthonormal completion).
  * **Solvers & Inverses:** Exact linear system solver (`solve`), least-squares solver (`lstsq`), matrix inverse (`inverse`), and Moore-Penrose pseudoinverse (`pinv`).
  * **Subspaces & Matrix Functions:** Matrix rank (`rank`), determinant (`det`), condition number (`cond`), orthonormal column space (`orth`), null space (`nullSpace`), Kronecker product (`kron`), and Padé scaling-and-squaring matrix exponential (`expm`).
  * **Data Science & Manifold Learning:** Covariance (`cov`), Pearson correlation (`corrcoef`), Principal Component Analysis (`pca`), and Multidimensional Scaling (`MDS_Classical`, `MDS_SMACOF`, `MDS_UCF`).

---

## Integration

Copy `code/SimpleMatrix.h` into your include path and `#include "SimpleMatrix.h"`.

* **For C++ projects (C++11 or newer):**
  ```bash
  g++ -std=c++11 -O2 your_program.cpp -o your_program
  ```
* **For Pure C projects (C99 / C11 or newer):**
  ```bash
  gcc -std=c11 -O2 your_program.c -lm -o your_program
  ```

---

## Quick Start

### C++ Quick Start

```cpp
#include <iostream>
#include "SimpleMatrix.h"

int main() {
    // 1. Construct matrices using initializer lists or static factories
    smat::Matrix<double> A = {
        {4.0, 12.0, -16.0},
        {12.0, 37.0, -43.0},
        {-16.0, -43.0, 98.0}
    };
    auto I = smat::Matrix<double>::eye(3);
    smat::Matrix<double> b = {{1.0}, {2.0}, {3.0}};

    // 2. Natural indexing and operator overloading
    A(0, 0) = 4.0;
    smat::Matrix<double> C = 2.0 * A + I;
    std::cout << "C = 2*A + I:\n" << C << "\n";

    // 3. Solve linear system A * x = b
    smat::Matrix<double> x = A.solve(b);
    std::cout << "Solution x to A*x = b:\n" << x << "\n";

    // 4. Matrix decompositions (Cholesky, SVD, Eigen, QR, LU)
    smat::Matrix<double> L = A.cholesky();
    smat::Matrix<double> U, S, Vt;
    A.svd(U, S, Vt);
    std::cout << "Singular values:\n" << S.diag().transpose() << "\n";

    return 0;
}
```

### Pure C Quick Start

```c
#include <stdio.h>
#include "SimpleMatrix.h"

int main(void) {
    // 1. Create matrices from array or factory functions
    double a_vals[9] = {
         4.0,  12.0, -16.0,
        12.0,  37.0, -43.0,
       -16.0, -43.0,  98.0
    };
    double b_vals[3] = {1.0, 2.0, 3.0};

    smat_t *A = smat_from_array(3, 3, a_vals);
    smat_t *b = smat_from_array(3, 1, b_vals);

    // 2. Solve A * x = b
    smat_t *x = smat_solve(A, b);
    printf("Solution x:\n");
    smat_print(x);

    // 3. Compute SVD: A = U * S * Vt
    smat_t *U = NULL, *S = NULL, *Vt = NULL;
    smat_svd(A, &U, &S, &Vt);
    printf("Largest singular value: %.4f\n", S->v[0][0]);

    // 4. Free allocated memory
    smat_free(A);
    smat_free(b);
    smat_free(x);
    smat_free(U);
    smat_free(S);
    smat_free(Vt);
    return 0;
}
```

---

## C++ API Reference (`smat::Matrix<T>`)

All C++ classes and functions reside in the `smat` namespace. `Matrix<T>` supports `double`, `float`, and `int`.

### Constructors & Factory Methods
* `Matrix()`: Default empty matrix ($0 \times 0$).
* `Matrix(int rows, int cols)`: Zero-initialized matrix of size `rows` $\times$ `cols`.
* `Matrix(int rows, int cols, T value)`: Matrix filled with `value`.
* `Matrix(std::initializer_list<std::initializer_list<T>> list)`: 2D initializer list constructor.
* `Matrix(int rows, int cols, std::string type)`: Special matrix constructor:
  * `"I"`, `"eye"`, `"identity"`: Identity matrix.
  * `"zeros"`, `"ones"`: All zeros or all ones.
  * `"rand"`: Uniform random values in $[0, 1)$.
  * `"randn"`: Standard normal $\mathcal{N}(0, 1)$ random values.
  * `"rand_int"`: Random non-negative integers.
  * `"randperm"`: Each column is a random permutation of $1 \dots \text{rows}$.
* `Matrix(const char* filename)` / `Matrix(const std::string& filename)`: Load whitespace/comma/tab-delimited matrix from text file.
* **Static Factory Methods:**
  * `Matrix<T>::zeros(int rows, int cols)`
  * `Matrix<T>::ones(int rows, int cols)`
  * `Matrix<T>::eye(int n)` / `Matrix<T>::eye(int rows, int cols)` / `Matrix<T>::identity(int n)`
  * `Matrix<T>::rand(int rows, int cols)` / `Matrix<T>::randn(int rows, int cols)`
  * `Matrix<T>::diag(const Matrix<T>& v)` / `Matrix<T>::diag(const std::vector<T>& v)`
  * `Matrix<T>::linspace(T start, T end, int n)`
  * `Matrix<T>::arange(T start, T stop, T step = 1)`
  * `Matrix<T>::fromVector(int rows, int cols, const std::vector<T>& vec)`
  * `Matrix<T>::fromArray(int rows, int cols, const T* arr)`
* `smat::seed(unsigned int s)`: Seed the random number generator for reproducible experiments.

### Element Access, Indexing & Properties
* `M(r, c)` / `M.get(r, c)` / `M.set(r, c, val)`: Bounds-checked 2D element access.
* `M(idx)`: Bounds-checked 1D flat row-major element access.
* `M[r][c]` / `M.v[r][c]`: Fast direct row-pointer access.
* `M.data()`: Pointer to contiguous underlying row-major array (`T*`).
* `M.rows()`, `M.columns()` / `M.cols()`, `M.size()`, `M.empty()`.
* `M.isSquare()`, `M.isSymmetric(double tol = 1e-9)`, `M.equals(other, double tol = 1e-9)`.
* `M.fill(T value)`, `M.copy()`, `M.print()`, `M.saveTxt(filename)`.
* `std::cout << M`: Formatted stream output.

### Shape, Slicing & Concatenation
* `M.transpose()` / `M.T_mat()`: Matrix transpose $M^T$.
* `M.reshape(int new_rows, int new_cols)`: Reshape without copying order of elements.
* `M.flatten()`: Reshape into a $(r \cdot c) \times 1$ column vector.
* `M.sub(int r1, int r2, int c1, int c2)`: Extract submatrix `M(r1:r2, c1:c2)` (inclusive, 0-indexed).
* `M.row(int r)` / `M.col(int c)`: Extract single row ($1 \times n$) or column ($m \times 1$).
* `M.setRow(int r, const Matrix<T>& v)` / `M.setCol(int c, const Matrix<T>& v)`: Replace row or column.
* `M.swapRows(int r1, int r2)` / `M.swapCols(int c1, int c2)`: In-place row/column swap.
* `M.diag()`: Extract diagonal as column vector (if $M$ is a matrix), or construct diagonal matrix (if $M$ is a row/column vector).
* `M.triu(int k = 0)` / `M.tril(int k = 0)`: Upper or lower triangular part.
* `M.concatenateRight(A)` / `M.hstack(A)`: Horizontal concatenation $[M, A]$.
* `M.concatenateBottom(A)` / `M.vstack(A)`: Vertical concatenation $[M; A]$.

### Operators, Arithmetic & Math Functions
* **Matrix Operators:** `A + B`, `A - B`, `A * B` (matrix product), `A += B`, `A -= B`, `A *= B`, `A == B`, `A != B`, `-A`.
* **Scalar Operators:** `A + s`, `s + A`, `A - s`, `s - A`, `A * s`, `s * A`, `A / s`, `A += s`, `A -= s`, `A *= s`, `A /= s`.
* **In-place Scalar Methods:** `addNumberSelf`, `subtractNumberSelf`, `multiplyNumberSelf`, `divideNumberSelf`.
* **Element-wise Matrix Methods:** `addMatrixSelf/New`, `subtractMatrixSelf/New`, `dotMultiplyMatrixSelf/New` ($A \odot B$), `dotDivideMatrixSelf/New` ($A \oslash B$), `multiplyMatrixNew` ($AB$).
* **Element-wise Functions:** `abs()`, `exp()`, `log()`, `sqrt()`, `power(p)`, `sin()`, `cos()`, `tanh()`, `apply(fn)`.

### Vector & Matrix Products
* `A.dot(B)`: Inner product $\sum_i A_i B_i$.
* `a.cross(b)`: 3D vector cross product $a \times b$.
* `a.outer(b)`: Outer product $a b^T$.
* `A.kron(B)`: Kronecker tensor product $A \otimes B$.

### Norms, Reductions & Statistics
* `sum()`, `mean()`, `var(bool sample = false)`, `std(bool sample = false)`.
* `minEl()`, `minEl(int &r, int &c)`, `maxEl()`, `maxEl(int &r, int &c)`.
* `sumAxis(int axis)`, `meanAxis(int axis)` (`axis = 0` for column-wise $1 \times n$, `axis = 1` for row-wise $m \times 1$).
* `trace()`: Sum of diagonal elements $\text{tr}(A)$.
* `fnorm()`: Frobenius norm $\|A\|_F$.
* `pnorm(double p)`: Entrywise $L_p$ norm.
* `norm1()`: Induced 1-norm (maximum absolute column sum).
* `normInf()`: Induced $\infty$-norm (maximum absolute row sum).
* `norm2()`: Spectral norm ($\sigma_{\max}(A)$).
* `cond()`: 2-norm condition number $\sigma_{\max}(A) / \sigma_{\min}(A)$.
* `cov()`: Sample covariance matrix ($p \times p$ for $n \times p$ data matrix).
* `corrcoef()`: Pearson correlation coefficient matrix ($p \times p$).

### Linear Algebra Decompositions & Solvers
* `lu(L, U, P)`: LU decomposition with partial pivoting ($PA = LU$). Works even on singular matrices.
* `cholesky()` / `cholesky(L)`: Cholesky factorization ($A = L L^T$) for symmetric positive-definite matrices.
* `qr(Q, R)`: Householder QR decomposition ($A = QR$) for any $m \times n$ matrix ($m \ge n$ or $m < n$).
* `eigen(V, D)` / `eigenvalues()`: Jacobi symmetric eigenvalue decomposition ($AV = VD$) with eigenvalues sorted in descending order.
* `svd(U, S, Vt)`: Thin Singular Value Decomposition ($A = USV^T$) with singular values sorted descending and guaranteed orthonormal $U, V$ even for rank-deficient matrices.
* `determinant()` / `det()`: Matrix determinant $\det(A)$.
* `rank(double tol = -1.0)`: Numerical rank via SVD.
* `inverse()` / `inv()`: Matrix inverse $A^{-1}$.
* `pinv(double tol = -1.0)`: Moore-Penrose pseudoinverse $A^+$ via SVD.
* `solve(B)`: Solve square linear system $AX = B$ via $PA = LU$.
* `lstsq(B)`: Minimum-norm least-squares solver $\min_X \|AX - B\|_F$ for overdetermined, underdetermined, or rank-deficient systems.
* `orth(double tol = -1.0)`: Orthonormal basis for the column space (range) of $A$.
* `nullSpace(double tol = -1.0)`: Orthonormal basis for the null space (kernel) of $A$.
* `expm()`: Matrix exponential $e^A$ via scaling and squaring Padé(6,6) approximant.

### Dimensionality Reduction (PCA & MDS)
* `X.pca(n_components, projected, components, explained_variance)`: Principal Component Analysis on data matrix $X$.
* `smat::EuclideanDistanceMatrix(X, D)`: Pairwise Euclidean distance matrix $D_{ij} = \|x_i - x_j\|_2$.
* `smat::MDS_Classical(D, dim)`: Classical (Torgerson) metric MDS via double centering and spectral decomposition.
* `smat::MDS_SMACOF(D, X0, dim, iter)`: Iterative SMACOF majorization MDS.
* `smat::MDS_UCF(D, X0, dim, iter)`: Iterative stochastic gradient MDS.

---

## Pure C API Reference (`smat_t`)

When included in a `.c` file (or via `extern "C"` in C++), `SimpleMatrix.h` exposes the `smat_t` (alias `SMatrix`) struct and `smat_*` functions.

### Types & Error Handling
```c
typedef struct {
    int rows;
    int cols;
    double *data; /* Contiguous row-major array of size rows * cols */
    double **v;   /* Row pointers: A->v[r][c] accesses row r, col c */
} smat_t;
```
Error status codes (`smat_err_t`): `SMAT_OK` (`0`), `SMAT_ERR_NULL`, `SMAT_ERR_ALLOC`, `SMAT_ERR_DIM`, `SMAT_ERR_BOUNDS`, `SMAT_ERR_SINGULAR`, `SMAT_ERR_NOT_SYMMETRIC`, `SMAT_ERR_NOT_POS_DEF`, `SMAT_ERR_IO`, `SMAT_ERR_INVALID_ARG`. Convert to human-readable text with `smat_strerror(err)`.

### Creation & Memory Management
* `smat_create(rows, cols)` / `smat_zeros(rows, cols)`
* `smat_create_val(rows, cols, val)` / `smat_ones(rows, cols)`
* `smat_eye(rows, cols)` / `smat_identity(n)`
* `smat_rand(rows, cols)` / `smat_randn(rows, cols)` / `smat_randperm(rows, cols)` / `smat_seed(seed)`
* `smat_from_array(rows, cols, arr)` / `smat_copy(A)`
* `smat_linspace(start, end, n)` / `smat_arange(start, stop, step)` / `smat_diag(A)`
* `smat_free(A)` (safe to call on `NULL`)

### Accessors, Slicing & File I/O
* `smat_rows(A)`, `smat_cols(A)`, `smat_size(A)`, `smat_get(A, r, c)`, `smat_set(A, r, c, val)`, `smat_fill(A, val)`
* `smat_is_square(A)`, `smat_is_symmetric(A, tol)`, `smat_equals(A, B, tol)`
* `smat_transpose(A)`, `smat_reshape(A, r, c)`, `smat_sub(A, r1, r2, c1, c2)`, `smat_row(A, r)`, `smat_col(A, c)`
* `smat_set_row(A, r, vec)`, `smat_set_col(A, c, vec)`, `smat_swap_rows(A, r1, r2)`, `smat_swap_cols(A, c1, c2)`
* `smat_concat_right(A, B)`, `smat_concat_bottom(A, B)`, `smat_triu(A, k)`, `smat_tril(A, k)`
* `smat_print(A)`, `smat_load_txt(filename)`, `smat_save_txt(A, filename)`

### Arithmetic, Products & Statistics
* Scalar: `smat_add_scalar`, `smat_mul_scalar`, `smat_add_scalar_self`, `smat_sub_scalar_self`, `smat_mul_scalar_self`, `smat_div_scalar_self`
* Element-wise: `smat_abs`, `smat_neg`, `smat_exp`, `smat_log`, `smat_sqrt`, `smat_pow`, `smat_apply`
* Matrix-Matrix: `smat_add`, `smat_sub_mat`, `smat_dot_mul`, `smat_dot_div` (and `*_self` in-place variants)
* Products: `smat_mul(A, B)`, `smat_dot(a, b)`, `smat_cross(a, b)`, `smat_outer(a, b)`, `smat_kron(A, B)`
* Statistics & Norms: `smat_sum`, `smat_mean`, `smat_var`, `smat_std`, `smat_std_sample`, `smat_trace`, `smat_fnorm`, `smat_pnorm`, `smat_norm_1`, `smat_norm_inf`, `smat_norm_2`, `smat_cond`, `smat_max`, `smat_min`, `smat_sum_axis`, `smat_mean_axis`, `smat_cov`, `smat_corrcoef`

### Linear Algebra, PCA & MDS in C
* `smat_lu(A, &L, &U, &P)`
* `smat_cholesky(A)`
* `smat_qr(A, &Q, &R)`
* `smat_eigen(A, &V, &D)` / `smat_eigenvalues(A)`
* `smat_svd(A, &U, &S, &Vt)`
* `smat_determinant(A)`, `smat_rank(A)`, `smat_inverse(A)`, `smat_pinv(A)`
* `smat_solve(A, B)`, `smat_lstsq(A, B)`, `smat_orth(A)`, `smat_null_space(A)`, `smat_expm(A)`
* `smat_pca(X, n_components, &projected, &components, &explained_variance)`
* `smat_pdist(X)`, `smat_euclidean_distance(X, D)`
* `smat_mds_classical(D, dim)`, `smat_mds_smacof(D, X0, dim, iter)`, `smat_mds_ucf(D, X0, dim, iter)`

---

## Examples & Demos

The `code/` directory includes demos and focused examples for both C++ and Pure C:
* `code/demo_Matrix.cpp`: General C++ matrix operations and Swiss Roll MDS benchmark.
* `code/demo_MatrixC.c`: Pure C walkthrough of matrix creation, Cholesky, linear solvers, eigendecomposition, SVD, and Classical MDS.
* `code/examples/example_linear_systems.cpp`: Linear systems (`solve`), LU decomposition, Cholesky decomposition, and ODE rotation via matrix exponential (`expm`).
* `code/examples/example_decompositions.cpp`: Rectangular QR decomposition, SVD low-rank matrix approximation, and fundamental subspaces (`orth`, `nullSpace`).
* `code/examples/example_data_science.cpp`: Quadratic polynomial regression (`lstsq`), covariance/correlation matrices (`cov`, `corrcoef`), PCA (`pca`), and Classical MDS (`MDS_Classical`).
* `code/examples/example_c_usage.c`: Pure C scientific computing with linear regression (`smat_lstsq`), continuous-time Markov chain matrix exponential (`smat_expm`), pseudoinverse (`smat_pinv`), and null space (`smat_null_space`).

---

## Building and Testing

Inside the `code/` directory, use the provided `Makefile`:

* **Build Everything (C & C++ tests, demos, and examples):**
  ```bash
  cd code && make
  ```
* **Run Unit Tests (C++ and Pure C test suites):**
  ```bash
  cd code && make test
  ```
* **Run Demos (C++ and Pure C demos):**
  ```bash
  cd code && make run
  ```
* **Run All Examples:**
  ```bash
  cd code && make examples
  ```
* **Clean Build Artifacts:**
  ```bash
  cd code && make clean
  ```

---

## Citation

If you find this library useful for your research, please cite:

```bibtex
@inproceedings{wang2013feature,
  title={Feature learning by multidimensional scaling and its applications in object recognition},
  author={Wang, Quan and Boyer, Kim L},
  booktitle={Graphics, Patterns and Images (SIBGRAPI), 2013 26th SIBGRAPI-Conference on},
  pages={8--15},
  year={2013},
  organization={IEEE}
}
```

## Copyright

```text
Copyright (C) 2013-2026 Quan Wang <wangq10@rpi.edu>
Signal Analysis and Machine Perception Laboratory
Department of Electrical, Computer, and Systems Engineering
Rensselaer Polytechnic Institute, Troy, NY 12180, USA
```