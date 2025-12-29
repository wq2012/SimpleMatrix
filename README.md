# SimpleMatrix [![C/C++ CI](https://github.com/wq2012/SimpleMatrix/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/wq2012/SimpleMatrix/actions/workflows/c-cpp.yml)

## About
SimpleMatrix is an extremely lightweight C++ matrix library, consisting of a single header file `SimpleMatrix.h`. It is designed for simplicity and ease of integration.

**Key Features:**
*   **Single-Header:** Easy to integrate; just drop `SimpleMatrix.h` into your project.
*   **Template Class:** Supports varying precision (e.g., `float`, `double`).
*   **Core Operations:** Implements essential matrix operations like addition, subtraction, multiplication, transposition, and submatrix extraction.
*   **Memory Safe:** Handles memory allocation and deallocation automatically (RAII compliant copy/assignment).
*   **Advanced Algorithms:** Includes LU, QR, Eigenvalue (Symmetric), SVD decompositions, Determinant, Inverse, and Linear Solver.
*   **MDS Algorithm:** Includes Multidimensional Scaling (MDS) algorithms (SMACOF and UCF versions).
*   **Modern C++ (C++11):** Supports Move Semantics and Initializer Lists for efficient and convenient usage.
*   **Robust:** Uses standard C++ exceptions for error handling.

**Limitations:**
*   Eigenvalue decomposition is currently implemented for symmetric matrices only.
*   Requires a C++11 compliant compiler.

## Integration
Simply include `SimpleMatrix.h` in your project. Ensure you compile with C++11 support:
```bash
g++ -std=c++11 your_code.cpp -o your_program
```

The library uses the `smat` namespace.

## Quick Start

```cpp
#include <iostream>
#include "SimpleMatrix.h"

int main() {
    // Initialize a 3x3 matrix with value 1.0
    // Initialize with C++11 list
    smat::Matrix<double> A = {{1, 2}, {3, 4}};

    // Or with size
    smat::Matrix<double> B(3, 3);
    B.set(0, 0, 1.0);
    
    // Set an element
    A.set(0, 0, 5.0);
    
    // Create an Identity matrix
    smat::Matrix<double> I(3, 3, "I");
    
    // Matrix Addition
    smat::Matrix<double> C = A + I;
    
    // Print result
    C.print();
    
    return 0;
}
```

## API Reference

### Constructors
*   `Matrix(int rows, int cols)`: Create uninitialized matrix.
*   `Matrix(int rows, int cols, T value)`: Create matrix filled with `value`.
*   `Matrix(int rows, int cols, std::string type)`: Create special matrix.
    *   `"I"`: Identity matrix.
    *   `"rand"`: Random values between 0 and 1.
    *   `"rand_int"`: Random integers.
    *   `"randperm"`: Random permutation.
*   `Matrix(const char* filename)`: Load matrix from a text file.

### Accessors
*   `void set(int r, int c, T value)`: Set element at (r, c).
*   `T get(int r, int c)`: Get element at (r, c).
*   `int rows()`: Get number of rows.
*   `int columns()`: Get number of columns.
*   `void print()`: Print matrix to variables formatted output.
*   `void saveTxt(const char* filename)`: Save matrix to text file.
*   `Matrix* copy()`: Return a pointer to a deep copy of the matrix.

### Basic Operations
*   `Matrix* transpose()`: Return a new transposed matrix.
*   `Matrix* sub(int r1, int r2, int c1, int c2)`: Return submatrix from (r1,c1) to (r2,c2).
*   `Matrix* concatenateRight(Matrix* A)`: Concatenate `A` to the right.
*   `Matrix* concatenateBottom(Matrix* A)`: Concatenate `A` to the bottom.

### Arithmetic & Math
All methods ending with `Self` modify the instance in-place. Methods ending with `New` or operator overloads return a new `Matrix` object.

*   **Operators:** `+`, `-`, `*` (Matrix Multiplication).
*   **Scalar:** `addNumberSelf`, `subtractNumberSelf`, `multiplyNumberSelf`, `divideNumberSelf`.
*   **Element-wise Matrix:** `addMatrixSelf/New`, `subtractMatrixSelf/New`, `dotMultiplyMatrixSelf/New`, `dotDivideMatrixSelf/New`.
*   **Math Functions:** `exp()`, `log()`, `sqrt()`, `power(p)`.

### Statistics
*   `T sum()`: Sum of all elements.
*   `double mean()`: Mean of all elements.
*   `double std()`: Standard deviation.
*   `T minEl(int& r, int& c)`: Minimum element (stores indices in r, c).
*   `T maxEl(int& r, int& c)`: Maximum element (stores indices in r, c).
*   `T trace()`: Trace of the matrix.
*   `double fnorm()`: Frobenius norm.
*   `double pnorm(double p)`: p-norm.

### Algorithms
*   `void lu(Matrix*& L, Matrix*& U, Matrix*& P)`: LU Decomposition ($PA = LU$).
*   `void qr(Matrix*& Q, Matrix*& R)`: QR Decomposition ($A = QR$).
*   `void eigen(Matrix*& V, Matrix*& D)`: Eigenvalue Decomposition for symmetric matrices ($A V = V D$).
*   `void svd(Matrix*& U, Matrix*& S, Matrix*& Vt)`: Singular Value Decomposition ($A = U S V^T$).
*   `T determinant()`: Compute determinant.
*   `int rank()`: Compute rank (number of singular values > threshold).
*   `Matrix* inverse()`: Compute inverse matrix.
*   `Matrix* solve(Matrix* B)`: Solve linear system $Ax=B$.
*   `MDS_SMACOF(...)`: Multidimensional Scaling using SMACOF.
*   `MDS_UCF(...)`: Multidimensional Scaling using UCF method.

## Building and Testing

A `Makefile` is provided for easy compilation and testing.

### Prerequisites
*   `g++` (GNU C++ Compiler)
*   `make`

### Build Commands
*   **Build Everything**:
    ```bash
    make
    ```
    This builds `demo_Matrix.exe` and `SimpleMatrixTest.exe`. Note: This command always cleans previous builds first.

*   **Run Demo**:
    ```bash
    make run
    ```
    Builds and runs the `demo_Matrix.exe`, demonstrating basic usage and MDS on the Swiss Roll dataset.

*   **Run Tests**:
    ```bash
    make test
    ```
    Builds and runs the unit tests in `SimpleMatrixTest.cpp`.

*   **Clean Artifacts**:
    ```bash
    make clean
    ```
    Removes all executables.

## Citation

If you find this library useful for your research, please cite:

```
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

```
Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>
Signal Analysis and Machine Perception Laboratory
Department of Electrical, Computer, and Systems Engineering
Rensselaer Polytechnic Institute, Troy, NY 12180, USA
```