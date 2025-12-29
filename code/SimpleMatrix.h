/***************************************************************
*  Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>
*  Signal Analysis and Machine Perception Laboratory
*  Department of Electrical, Computer, and Systems Engineering
*  Rensselaer Polytechnic Institute, Troy, NY 12180, USA
*
*  Related publication:
*  Quan Wang, Kim L. Boyer.
*  Feature Learning by Multidimensional Scaling and its Applications in Object Recognition.
*  2013 26th SIBGRAPI Conference on Graphics, Patterns and Images (Sibgrapi). IEEE, 2013.
***************************************************************/

#ifndef SIMPLE_MATRIX_H
#define SIMPLE_MATRIX_H

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <initializer_list> // Kept from original, needed for initializer_list constructor
#include <utility>          // Kept from original, potentially needed
#include <ctime>            // Kept from original, potentially needed

#define MAX_LINE_LENGTH 100000
#define EPSILON 0.0000001
namespace smat
{

/**********************************************
  * Declaration part
  **********************************************/

template <class T>
class Matrix
{
public:
  // Data members
  // v acts as row pointers for convenient [i][j] access
  std::vector<T*> v; 
  
private:
  int rows_;
  int columns_;
  std::vector<T> data_; // Contiguous data storage

public:
  // Constructors
  Matrix();                                      // default constructor
  Matrix(int rows, int columns);                 // constructor with size
  Matrix(int rows, int columns, T value);        // constructor with size and value
  Matrix(int rows, int columns, std::string type); // constructor with size and random/identity/type
  
  Matrix(const Matrix &other);                   // copy constructor
  Matrix(Matrix &&other);                        // move constructor
  Matrix(std::initializer_list<std::initializer_list<T>> list); // initializer list
  Matrix(const char *filename);                  // read from file

  // Assignment operators
  Matrix& operator=(const Matrix &other);        // copy assignment
  Matrix& operator=(Matrix &&other);             // move assignment

  // Destructor
  ~Matrix();                                     // destruction

  void set(int r, int c, T value); // row, column, value
  T get(int r, int c) const;             // row, column
  int rows() const;                      // number of rows
  int columns() const;                   // number of columns

  void print() const;   // print the matrix
  Matrix<T> copy() const; // copy itself to a new matrix

  void saveTxt(const char *filename) const; // save matrix to txt file

  // B=M'
  Matrix<T> transpose() const;
  // B=M(r1:r2,c1:c2)
  Matrix<T> sub(int r1, int r2, int c1, int c2) const; // submatrix
  // B=|M|
  Matrix<T> abs() const; // absolute values

  // numbers of matrix
  T trace() const;               // trace
  double fnorm() const;          // Frobenius norm
  double pnorm(double p) const;  // p-norm
  T maxEl(int &r, int &c) const; // max element
  T minEl(int &r, int &c) const; // min element
  double mean() const;           // mean of elements
  T sum() const;                 // sum of elements
  double std() const;            // standard deviation of elements

  // M=M+a
  void addNumberSelf(T value); // add a number to itself in space
  // M=M-a
  void subtractNumberSelf(T value); // subtract a number from itself in space
  // M=M*a
  void multiplyNumberSelf(T value); // multiply a number to itself in space
  // M=M/a
  void divideNumberSelf(T value); // divide itself by a number in space

  // Element-wise functions
  Matrix<T> exp() const;
  Matrix<T> log() const;
  Matrix<T> sqrt() const;
  Matrix<T> power(double p) const;

  // M=M+A
  void addMatrixSelf(const Matrix<T> &A); // add a matrix to itself in space
  // M=M-A
  void subtractMatrixSelf(const Matrix<T> &A); // subtract a matrix from itself in space
  // M=M.*A
  void dotMultiplyMatrixSelf(const Matrix<T> &A); // dot multiply a matrix to itself in space
  // M=M./A
  void dotDivideMatrixSelf(const Matrix<T> &A); // dot divide a matrix to itself in space

  // B=M+A
  Matrix<T> addMatrixNew(const Matrix<T> &A) const; // add a matrix to itself with new matrix
  // B=M-A
  Matrix<T> subtractMatrixNew(const Matrix<T> &A) const; // subtract a matrix from itself with new matrix
  // B=M.*A
  Matrix<T> dotMultiplyMatrixNew(const Matrix<T> &A) const; // dot multiply a matrix to itself with new matrix
  // B=M./A
  Matrix<T> dotDivideMatrixNew(const Matrix<T> &A) const; // dot divide a matrix to itself with new matrix
  // B=M*A
  Matrix<T> multiplyMatrixNew(const Matrix<T> &A) const; // multiply a matrix to itself with new matrix

  // Concatenation
  Matrix<T> concatenateRight(const Matrix<T> &A) const;
  Matrix<T> concatenateBottom(const Matrix<T> &A) const;

  // Linear Algebra
  void lu(Matrix<T> &L, Matrix<T> &U, Matrix<T> &P) const;   // LU Decomposition
  void qr(Matrix<T> &Q, Matrix<T> &R) const;               // QR Decomposition
  void eigen(Matrix<T> &V, Matrix<T> &D) const;            // Eigenvalue Decomposition
  void svd(Matrix<T> &U, Matrix<T> &S, Matrix<T> &Vt) const; // Singular Value Decomposition
  T determinant() const;                               // Determinant
  int rank() const;                                    // Matrix Rank
  Matrix<T> inverse() const;                             // Matrix Inverse
  Matrix<T> solve(const Matrix<T> &B) const;                      // Solve Ax=B

  // Operator overloading
  Matrix<T> operator+(const Matrix<T> &other) const;
  Matrix<T> operator-(const Matrix<T> &other) const;
  Matrix<T> operator*(const Matrix<T> &other) const;
};

/**********************************************
  * Utilities part
  **********************************************/

template <class T>
T min(T v1, T v2)
{
  if (v1 < v2)
    return v1;
  else
    return v2;
}

template <class T>
T max(T v1, T v2)
{
  if (v1 > v2)
    return v1;
  else
    return v2;
}

template <class T>
void swap(T &v1, T &v2)
{
  T v3 = v1;
  v1 = v2;
  v2 = v3;
}

template <class T>
double sign(T v)
{
  if (v > 0)
    return 1.0;
  else if (v < 0)
    return -1.0;
  else
    return 0.0;
}

/**********************************************
  * Implementation part
  **********************************************/

template <class T>
Matrix<T>::Matrix() : rows_(0), columns_(0) // default constructor
{
}

template <class T>
Matrix<T>::Matrix(int rows, int columns) // constructor with size
{
  if (rows < 1 || columns < 1)
  {
    throw std::invalid_argument("Invalid construction arguments: rows=" + std::to_string(rows) + ", columns=" + std::to_string(columns));
  }

  rows_ = rows;
  columns_ = columns;

  data_.resize(rows * columns);
  v.resize(rows);
  for(int i = 0; i < rows; ++i) {
      v[i] = &data_[i * columns];
  }
}

template <class T>
Matrix<T>::Matrix(int rows, int columns, T value) // initialization with all same values
{
  if (rows < 1 || columns < 1)
  {
    throw std::invalid_argument("Invalid construction arguments: rows=" + std::to_string(rows) + ", columns=" + std::to_string(columns));
  }

  rows_ = rows;
  columns_ = columns;

  data_.assign(rows * columns, value);
  v.resize(rows);
  for(int i = 0; i < rows; ++i) {
      v[i] = &data_[i * columns];
  }
}

template <class T>
Matrix<T>::Matrix(int rows, int columns, std::string type) // special matrix such as I
{
  if (rows < 1 || columns < 1)
  {
    throw std::invalid_argument("Invalid construction arguments: rows=" + std::to_string(rows) + ", columns=" + std::to_string(columns));
  }
  rows_ = rows;
  columns_ = columns;

  data_.resize(rows * columns); // Default constructs to 0
  v.resize(rows);
  for(int i = 0; i < rows; ++i) {
      v[i] = &data_[i * columns];
  }

  if (type.compare("I") == 0)
  {
    for (int i = 0; i < std::min(rows_, columns_); i++)
    {
      v[i][i] = (T)1;
    }
  }
  else if (type.compare("rand") == 0) // all elements between 0 and 1
  {
    srand(time(NULL));
    for (int i = 0; i < rows_; i++)
    {
      for (int j = 0; j < columns_; j++)
      {
        v[i][j] = (T)rand() / RAND_MAX;
      }
    }
  }

  else if (type.compare("rand_int") == 0)
  {
    srand(time(NULL));
    for (int i = 0; i < rows_; i++)
    {
      for (int j = 0; j < columns_; j++)
      {
        v[i][j] = (T)rand();
      }
    }
  }

  else if (type.compare("randperm") == 0) // random permutation, each column is a randperm vector of size rows*1
  {
    srand(time(NULL));
    for(int j=0; j<columns; ++j) {
      std::vector<int> col(rows);
      for(int k=0; k<rows; ++k) col[k] = k + 1;
      std::random_shuffle(col.begin(), col.end());
      for(int i=0; i<rows; ++i) v[i][j] = (T)col[i];
    }
  }

  else
  {
    throw std::invalid_argument("Undefined matrix type: " + type);
  }
}

template <class T>
Matrix<T>::Matrix(Matrix &&other) : rows_(0), columns_(0) // move constructor
{
  // Swap resources
  std::swap(rows_, other.rows_);
  std::swap(columns_, other.columns_);
  std::swap(data_, other.data_);
  std::swap(v, other.v);
  
  // other is left in a valid empty/swapped state
}

template <class T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list) // initializer list
{
  rows_ = list.size();
  if (rows_ == 0)
  {
    columns_ = 0;
    return;
  }
  columns_ = list.begin()->size();

  data_.resize(rows_ * columns_);
  v.resize(rows_);
  for(int i = 0; i < rows_; ++i) {
      v[i] = &data_[i * columns_];
  }
  
  int i = 0;
  for (const auto &row : list)
  {
    if ((int)row.size() != columns_)
    {
        // Vector automatically cleans up, no manual cleanup needed
        throw std::invalid_argument("All rows in initializer list must have the same size.");
    }
    int j = 0;
    for (const auto &val : row)
    {
      v[i][j] = val;
      j++;
    }
    i++;
  }
}

template <class T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other) // move assignment operator
{
  if (this != &other)
  {
      rows_ = other.rows_;
      columns_ = other.columns_;
      data_ = std::move(other.data_);
      v = std::move(other.v);
      
      // Re-link v pointers to moved data_
      for(int i = 0; i < rows_; ++i) {
          v[i] = &data_[i * columns_];
      }
      
      other.rows_ = 0;
      other.columns_ = 0;
  }
  return *this;
}

template <class T>
Matrix<T>::Matrix(const char *filename)
{
  FILE *pFile;
  // first pass: matrix size
  int rows = 0;
  int columns = 0;

  pFile = fopen(filename, "r");
  if (pFile == NULL)
  {
    throw std::runtime_error(std::string("File \"") + filename + "\" cannot be found.");
  }
  char line[MAX_LINE_LENGTH];
  char *token = NULL;
  while (fgets(line, MAX_LINE_LENGTH, pFile) != NULL)
  {
    rows++;
    if (rows == 1) // count the number of columns
    {
      token = strtok(line, " ,\t");
      while (token != NULL && token[0] >= 32)
      {
        columns++;
        token = strtok(NULL, " ,\t");
      }
    }
    else // check whether every row contains the same number of elements with the first row
    {
      int check = 0;
      token = strtok(line, " ,\t");
      while (token != NULL && token[0] >= 32)
      {
        check++;
        token = strtok(NULL, " ,\t");
      }
      if (check < columns)
      {
        rows--;
        break;
      }
    }
  }
  fclose(pFile);
  printf("Reading matrix from file \"%s\": %d rows, %d columns\n", filename, rows, columns);

  // second pass: read data
  rows_ = rows;
  columns_ = columns;
  
  data_.resize(rows * columns);
  v.resize(rows);
  for(int i = 0; i < rows; ++i) {
      v[i] = &data_[i * columns];
  }


  pFile = fopen(filename, "r");
  if (pFile == NULL)
  {
     // Should not happen as we opened it before
     throw std::runtime_error("File cannot be found (second pass).");
  }
  int i = 0;
  while (fgets(line, MAX_LINE_LENGTH, pFile) != NULL)
  {
    if (i >= rows)
      break;
    for (int j = 0; j < columns; j++)
    {
      if (j == 0)
        token = strtok(line, " ,\t");
      else
        token = strtok(NULL, " ,\t");
      v[i][j] = (T)atof(token);
    }
    i++;
  }
  fclose(pFile);
}

template <class T>
Matrix<T>::Matrix(const Matrix &other) // copy constructor
{
  rows_ = other.rows_;
  columns_ = other.columns_;
  data_ = other.data_; // Vector copy
  v.resize(rows_);
  for(int i = 0; i < rows_; ++i) {
      v[i] = &data_[i * columns_];
  }
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) // assignment operator
{
  if (this != &other)
  {
      rows_ = other.rows_;
      columns_ = other.columns_;
      data_ = other.data_;
      
      v.resize(rows_);
      for(int i = 0; i < rows_; ++i) {
          v[i] = &data_[i * columns_];
      }
  }
  return *this;
}

template <class T>
Matrix<T>::~Matrix() // destruction
{
  // No manual memory releasing needed
}

template <class T>
void Matrix<T>::set(int r, int c, T value) // row, column, value
{
  if (r < 0 || r >= rows_ || c < 0 || c >= columns_)
  {
    throw std::out_of_range("Invalid index in set(): r=" + std::to_string(r) + ", c=" + std::to_string(c));
  }
  v[r][c] = value;
}

template <class T>
T Matrix<T>::get(int r, int c) const // row, column
{
  if (r < 0 || r >= rows_ || c < 0 || c >= columns_)
  {
    throw std::out_of_range("Matrix index out of range in get().");
  }
  return v[r][c];
}

template <class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const
{
  return this->addMatrixNew(other);
}

template <class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) const
{
  return this->subtractMatrixNew(other);
}

template <class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const
{
  return this->multiplyMatrixNew(other);
}


template <class T>
int Matrix<T>::rows() const // number of rows
{
  return rows_;
}

template <class T>
int Matrix<T>::columns() const // number of columns
{
  return columns_;
}

template <class T>
void Matrix<T>::print() const // print the matrix
{
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      if (std::abs((double)v[i][j]) < EPSILON)
        printf("%10.4f ", 0.0);
      else
        printf("%10.4f ", (double)v[i][j]);
    }
    printf("\n");
  }
}

template <class T>
Matrix<T> Matrix<T>::copy() const // copy itself to a new matrix
{
  Matrix<T> B(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B.v[i][j] = v[i][j];
    }
  }
  return B;
}

template <class T>
void Matrix<T>::saveTxt(const char *filename) const // save matrix to txt file
{
  FILE *pFile = fopen(filename, "w");
  if (pFile == NULL)
  {
    throw std::runtime_error("File cannot be opened for writing.");
  }
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      fprintf(pFile, "%f ", (double)v[i][j]);
    }
    fprintf(pFile, "\n");
  }
  fclose(pFile);
  printf("Matrix saved to file \"%s\"\n", filename);
}

template <class T>
Matrix<T> Matrix<T>::transpose() const
{
  Matrix<T> B(columns_, rows_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B.v[j][i] = v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::sub(int r1, int r2, int c1, int c2) const // submatrix
{
  if (r1 < 0 || r2 >= rows_ || c1 < 0 || c2 >= columns_ || r1 > r2 || c1 > c2)
  {
    throw std::out_of_range("Matrix index out of range in sub().");
  }

  Matrix<T> B(r2 - r1 + 1, c2 - c1 + 1);
  for (int i = r1; i <= r2; i++)
  {
    for (int j = c1; j <= c2; j++)
    {
      B.v[i - r1][j - c1] = v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::abs() const // absolute values
{
  Matrix<T> B(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B.v[i][j] = std::abs((double)v[i][j]);
    }
  }
  return B;
}

template <class T>
T Matrix<T>::trace() const // trace
{
  T sum = 0;
  for (int i = 0; i < std::min(rows_, columns_); i++)
    sum += v[i][i];
  return sum;
}

template <class T>
double Matrix<T>::fnorm() const // Frobenius norm
{
  double sum = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      sum += std::pow((double)v[i][j], 2);
    }
  }
  return std::sqrt(sum);
}

template <class T>
double Matrix<T>::pnorm(double p) const // p-norm
{
  double sum = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      sum += std::pow(std::fabs((double)v[i][j]), p);
    }
  }
  return std::pow(sum, 1.0 / p);
}

template <class T>
T Matrix<T>::maxEl(int &r, int &c) const // max element
{
  T max = v[0][0];
  r = 0;
  c = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      if (v[i][j] > max)
      {
        max = v[i][j];
        r = i;
        c = j;
      }
    }
  }
  return max;
}

template <class T>
T Matrix<T>::minEl(int &r, int &c) const // min element
{
  T min = v[0][0];
  r = 0;
  c = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      if (v[i][j] < min)
      {
        min = v[i][j];
        r = i;
        c = j;
      }
    }
  }
  return min;
}

template <class T>
double Matrix<T>::mean() const // mean of elements
{
  return (double)this->sum() / (double)(rows_ * columns_);
}

template <class T>
T Matrix<T>::sum() const // sum of elements
{
  T sum = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      sum += v[i][j];
    }
  }
  return sum;
}

template <class T>
double Matrix<T>::std() const // standard deviation
{
  double m = mean();
  double sum = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      sum += std::pow((double)v[i][j] - m, 2);
    }
  }
  return std::sqrt(sum / (double)(rows_ * columns_));
}

template <class T>
void Matrix<T>::addNumberSelf(T value) // add a number to itself in space
{
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      v[i][j] += value;
    }
  }
}

template <class T>
void Matrix<T>::subtractNumberSelf(T value)
{
  addNumberSelf(-value);
}

template <class T>
void Matrix<T>::multiplyNumberSelf(T value) // add a number to itself in space
{
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      v[i][j] *= value;
    }
  }
}

template <class T>
void Matrix<T>::divideNumberSelf(T value)
{
  multiplyNumberSelf(1.0 / value);
}

template <class T>
Matrix<T> Matrix<T>::exp() const
{
  Matrix<T> B(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B.v[i][j] = std::exp((double)v[i][j]);
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::log() const
{
  Matrix<T> B(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B.v[i][j] = std::log((double)v[i][j]);
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::sqrt() const
{
  Matrix<T> B(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B.v[i][j] = std::sqrt((double)v[i][j]);
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::power(double p) const
{
  Matrix<T> B(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B.v[i][j] = std::pow((double)v[i][j], p);
    }
  }
  return B;
}

template <class T>
void Matrix<T>::addMatrixSelf(const Matrix<T> &A) // add a matrix to itself in space
{
  if (rows_ != A.rows() || columns_ != A.columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix summation.");
  }

  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      v[i][j] += A.v[i][j];
    }
  }
}

template <class T>
void Matrix<T>::subtractMatrixSelf(const Matrix<T> &A)
{
  if (rows_ != A.rows() || columns_ != A.columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix subtraction.");
  }

  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      v[i][j] -= A.v[i][j];
    }
  }
}

template <class T>
void Matrix<T>::dotMultiplyMatrixSelf(const Matrix<T> &A) // dot multiply a matrix to itself in space
{
  if (rows_ != A.rows() || columns_ != A.columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix dot multiplication.");
  }

  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      v[i][j] *= A.v[i][j];
    }
  }
}

template <class T>
void Matrix<T>::dotDivideMatrixSelf(const Matrix<T> &A)
{
  if (rows_ != A.rows() || columns_ != A.columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix dot division.");
  }

  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      v[i][j] /= A.v[i][j];
    }
  }
}

template <class T>
Matrix<T> Matrix<T>::addMatrixNew(const Matrix<T> &A) const // add a matrix to itself with new matrix
{
  if (rows_ != A.rows() || columns_ != A.columns())
  {
    throw std::invalid_argument("Matrix dimensions must match for addition.");
  }

  Matrix<T> B(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B.v[i][j] = v[i][j] + A.v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::subtractMatrixNew(const Matrix<T> &A) const
{
  if (rows_ != A.rows() || columns_ != A.columns())
  {
    throw std::invalid_argument("Matrix dimensions must match for subtraction.");
  }

  Matrix<T> B(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B.v[i][j] = v[i][j] - A.v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::dotMultiplyMatrixNew(const Matrix<T> &A) const // dot multiply a matrix to itself with new matrix
{
  if (rows_ != A.rows() || columns_ != A.columns())
  {
    throw std::invalid_argument("Matrix dimensions must match for dot multiplication.");
  }

  Matrix<T> B(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B.v[i][j] = v[i][j] * A.v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::dotDivideMatrixNew(const Matrix<T> &A) const
{
  if (rows_ != A.rows() || columns_ != A.columns())
  {
    throw std::invalid_argument("Matrix dimensions must match for dot division.");
  }

  Matrix<T> B(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B.v[i][j] = v[i][j] / A.v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::multiplyMatrixNew(const Matrix<T> &A) const // multiply a matrix to itself with new matrix
{
  if (columns_ != A.rows())
  {
    throw std::invalid_argument("Matrix dimensions must match for multiplication.");
  }

  Matrix<T> B(rows_, A.columns());
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < A.columns(); j++)
    {
      B.v[i][j] = 0;
      for (int k = 0; k < columns_; k++)
      {
        B.v[i][j] += v[i][k] * A.v[k][j];
      }
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::concatenateRight(const Matrix<T> &A) const
{
  if (rows_ != A.rows())
  {
    throw std::invalid_argument("Matrix rows must match for concatenateRight().");
  }
  Matrix<T> B(rows_, columns_ + A.columns());
  for (int i = 0; i < rows_; i++)
  {
    for(int j=0; j<columns_; j++) B.v[i][j] = v[i][j];
    for(int j=0; j<A.columns(); j++) B.v[i][j + columns_] = A.v[i][j];
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::concatenateBottom(const Matrix<T> &A) const
{
  if (columns_ != A.columns())
  {
    throw std::invalid_argument("Matrix columns must match for concatenateBottom().");
  }
  Matrix<T> B(rows_ + A.rows(), columns_);
  for(int j=0; j<columns_; j++)
  {
    for(int i=0; i<rows_; i++) B.v[i][j] = v[i][j];
    for(int i=0; i<A.rows(); i++) B.v[i + rows_][j] = A.v[i][j];
  }
  return B;
}



template <class T>
void Matrix<T>::lu(Matrix<T> &L, Matrix<T> &U, Matrix<T> &P) const // LU Decomposition
{
  if (rows_ != columns_)
  {
    throw std::invalid_argument("LU decomposition requires square matrix.");
  }
  int n = rows_;
  L = Matrix<T>(n, n, 0.0);
  U = Matrix<T>(n, n, 0.0);
  P = Matrix<T>(n, n, "I");

  Matrix<T> A = this->copy(); // Work on a copy

  for (int i = 0; i < n; i++)
  {
    // Pivot
    T maxEl = std::abs((double)A.v[i][i]);
    int maxRow = i;
    for (int k = i + 1; k < n; k++)
    {
      if (std::abs((double)A.v[k][i]) > maxEl)
      {
        maxEl = std::abs((double)A.v[k][i]);
        maxRow = k;
      }
    }

    // Swap rows in A and P
    for (int k = 0; k < n; k++)
    {
      T tmp = A.v[i][k];
      A.v[i][k] = A.v[maxRow][k];
      A.v[maxRow][k] = tmp;

      tmp = P.v[i][k];
      P.v[i][k] = P.v[maxRow][k];
      P.v[maxRow][k] = tmp;
    }
    
    for (int k=0; k<i; k++) {
        T tmp = L.v[i][k];
        L.v[i][k] = L.v[maxRow][k];
        L.v[maxRow][k] = tmp;
    }
    
    L.v[i][i] = 1.0;

    for (int k = i + 1; k < n; k++)
    {
      double c = -A.v[k][i] / A.v[i][i];
      L.v[k][i] = -c; // Multiplier
      for (int j = i; j < n; j++)
      {
        if (i == j)
          A.v[k][j] = 0;
        else
          A.v[k][j] = A.v[k][j] + c * A.v[i][j];
      }
    }
  }

  // U is the remaining A
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      if (j >= i)
        U.v[i][j] = A.v[i][j];
    }
  }
}

template <class T>
T Matrix<T>::determinant() const // Determinant
{
  if (rows_ != columns_)
  {
    throw std::invalid_argument("Determinant requires square matrix.");
  }
  
  Matrix<T> A = this->copy();
  T det = 1.0;
  int n = rows_;
  
  for (int i = 0; i < n; i++)
  {
    int pivot = i;
    for (int j = i + 1; j < n; j++) {
      if (std::abs((double)A.v[j][i]) > std::abs((double)A.v[pivot][i]))
        pivot = j; 
    }
    
    if (std::abs((double)A.v[pivot][i]) < EPSILON) {
      return 0.0;
    }
    
    if (pivot != i) {
       // Swap rows
       for(int k=0; k<n; k++) {
          T tmp = A.v[i][k];
          A.v[i][k] = A.v[pivot][k];
          A.v[pivot][k] = tmp;
       }
       det = -det;
    }
    
    det *= A.v[i][i];
    
    for (int j = i + 1; j < n; j++) {
      double factor = A.v[j][i] / A.v[i][i];
      for (int k = i + 1; k < n; k++) {
         A.v[j][k] = A.v[j][k] - factor * A.v[i][k];
      }
    }
  }
  
  return det;
}

template <class T>
Matrix<T> Matrix<T>::inverse() const // Matrix Inverse
{
  if (rows_ != columns_)
  {
    throw std::invalid_argument("Inverse requires square matrix.");
  }
  
  int n = rows_;
  Matrix<T> Inv(n, n);
  Matrix<T> A = this->copy();
  
  // Initialize Inv as Identity
  for(int i=0; i<n; i++)
    for(int j=0; j<n; j++)
      Inv.v[i][j] = (i==j) ? 1.0 : 0.0;
      
  for (int i = 0; i < n; i++) {
      int pivot = i;
      for(int j=i+1; j<n; j++)
        if(std::abs((double)A.v[j][i]) > std::abs((double)A.v[pivot][i])) pivot = j;
        
      if(std::abs((double)A.v[pivot][i]) < EPSILON) {
          throw std::runtime_error("Matrix is singular, cannot find inverse.");
      }
      
      // Swap rows in A and Inv
      if(pivot != i) {
          for(int k=0; k<n; k++) {
              T tmp = A.v[i][k]; A.v[i][k] = A.v[pivot][k]; A.v[pivot][k] = tmp;
              tmp = Inv.v[i][k]; Inv.v[i][k] = Inv.v[pivot][k]; Inv.v[pivot][k] = tmp;
          }
      }
      
      // Scale row i to make diagonal 1
      double div = A.v[i][i];
      for(int k=0; k<n; k++) {
          A.v[i][k] /= div;
          Inv.v[i][k] /= div;
      }
      
      // Eliminate other rows
      for(int j=0; j<n; j++) {
          if(i != j) {
              double mul = A.v[j][i];
              for(int k=0; k<n; k++) {
                  A.v[j][k] -= mul * A.v[i][k];
                  Inv.v[j][k] -= mul * Inv.v[i][k];
              }
          }
      }
  }
  
  return Inv;
}

template <class T>
Matrix<T> Matrix<T>::solve(const Matrix<T> &B) const // Solve Ax=B
{
    if (rows_ != columns_)
    {
        throw std::invalid_argument("Matrix must be square to solve system.");
    }
    if (rows_ != B.rows())
    {
        throw std::invalid_argument("Matrix row dimensions must agree.");
    }

    Matrix<T> L, U, P;
    this->lu(L, U, P);

    // Ax = B -> LUx = PB
    // Let y = Ux, then Ly = PB
    
    // 1. Compute PB
    Matrix<T> PB = P.multiplyMatrixNew(B);
    
    // 2. Solve Ly = PB (Forward substitution)
    Matrix<T> y(rows_, B.columns());
    for(int k=0; k<B.columns(); k++)
    {
        for(int i=0; i<rows_; i++)
        {
            double sum = 0;
            for(int j=0; j<i; j++)
            {
                sum += (double)L.v[i][j] * y.v[j][k];
            }
            y.v[i][k] = (PB.v[i][k] - sum); // L diagonal is 1
        }
    }
    
    // 3. Solve Ux = y (Backward substitution)
    Matrix<T> x(rows_, B.columns());
    for(int k=0; k<B.columns(); k++)
    {
        for(int i=rows_-1; i>=0; i--)
        {
            double sum = 0;
            for(int j=i+1; j<columns_; j++)
            {
                sum += (double)U.v[i][j] * x.v[j][k];
            }
            if (std::abs((double)U.v[i][i]) < EPSILON)
            {
                throw std::runtime_error("Matrix is singular.");
            }
            x.v[i][k] = (y.v[i][k] - sum) / (double)U.v[i][i];
        }
    }
    
    return x;
}

template <class T>
void Matrix<T>::qr(Matrix<T> &Q, Matrix<T> &R) const // QR Decomposition
{
    // Householder transformations
    // A = Q R
    // R is upper triangular, Q is orthogonal
    if (rows_ < columns_) {
        throw std::invalid_argument("QR decomposition requires rows >= columns.");
    }
    
    int m = rows_;
    int n = columns_;
    
    Q = Matrix<T>(m, m, "I");
    R = this->copy();
    
    std::vector<double> v_vec(m);
    
    for (int k = 0; k < std::min(m-1, n); k++) {
        double norm_x = 0;
        for(int i=k; i<m; i++) norm_x += R.v[i][k] * R.v[i][k];
        norm_x = std::sqrt(norm_x);
        
        if(norm_x < EPSILON) continue;
        
        double u1 = R.v[k][k];
        double sign = (u1 >= 0) ? 1.0 : -1.0;
        
        // Construct v
        v_vec[k] = u1 + sign * norm_x;
        for(int i=k+1; i<m; i++) v_vec[i] = R.v[i][k];
        
        double norm_v_sq = 0;
        for(int i=k; i<m; i++) norm_v_sq += v_vec[i]*v_vec[i];
        
        if (norm_v_sq < EPSILON) continue;
        double tau = 2.0 / norm_v_sq;
        
        // Update R: R = H R
        for (int j = k; j < n; j++) {
            double dot = 0;
            for(int i=k; i<m; i++) dot += v_vec[i] * R.v[i][j];
             
            for(int i=k; i<m; i++) {
                R.v[i][j] = R.v[i][j] - tau * dot * v_vec[i];
            }
        }
        
        // Update Q: Q = Q H
        for (int i=0; i<m; i++) {
             double dot = 0;
             for(int j=k; j<m; j++) dot += Q.v[i][j] * v_vec[j];
             
             for(int j=k; j<m; j++) {
                 Q.v[i][j] = Q.v[i][j] - tau * dot * v_vec[j];
             }
        }
    }
}

template <class T>
void Matrix<T>::eigen(Matrix<T> &V, Matrix<T> &D) const // Eigenvalue Decomposition
{
    // Jacobi eigenvalue algorithm to find eigenvalues and eigenvectors of a real symmetric matrix
    // A V = V D
    if (rows_ != columns_)
    {
      throw std::invalid_argument("Eigen decomposition requires square matrix.");
    }
    
    int n = rows_;
    D = this->copy(); // Will become diagonal
    V = Matrix<T>(n, n, "I"); // Accumulate rotations
    
    int maxRotations = 5 * n * n;
    
    for (int i = 0; i < maxRotations; i++) {
        // Find largest off-diagonal element
        double maxEl = 0;
        int p = -1, q = -1;
        
        for (int r = 0; r < n-1; r++) {
            for (int c = r+1; c < n; c++) {
                if (std::abs((double)D.v[r][c]) > maxEl) {
                    maxEl = std::abs((double)D.v[r][c]);
                    p = r; q = c;
                }
            }
        }
        
        if (maxEl < EPSILON) break; // Converged
        
        double App = D.v[p][p];
        double Aqq = D.v[q][q];
        double Apq = D.v[p][q];
        
        double theta = 0.5 * std::atan2(2 * Apq, Aqq - App);
        double c = std::cos(theta);
        double s = std::sin(theta);
        
        D.v[p][p] = c*c*App - 2*s*c*Apq + s*s*Aqq;
        D.v[q][q] = s*s*App + 2*s*c*Apq + c*c*Aqq;
        D.v[p][q] = 0.0;
        D.v[q][p] = 0.0;
        
        for (int k=0; k<n; k++) {
            if (k!=p && k!=q) {
                double Dpk = D.v[p][k];
                double Dqk = D.v[q][k];
                D.v[p][k] = c*Dpk - s*Dqk;
                D.v[k][p] = D.v[p][k]; // symmetry
                
                D.v[q][k] = s*Dpk + c*Dqk;
                D.v[k][q] = D.v[q][k];
            }
        }
        
        // Update V (V' = V J)
        for (int k=0; k<n; k++) {
            double Vkp = V.v[k][p];
            double Vkq = V.v[k][q];
            V.v[k][p] = c*Vkp - s*Vkq;
            V.v[k][q] = s*Vkp + c*Vkq;
        }
    }
}



template <class T>
void Matrix<T>::svd(Matrix<T> &U, Matrix<T> &S, Matrix<T> &Vt) const // Singular Value Decomposition
{
    // One-sided Jacobi SVD
    // A = U S Vt
    
    // Handle m < n by recursion on A^T
    if (rows_ < columns_) {
        Matrix<T> At = this->transpose();
        Matrix<T> U_t, S_t, Vt_t;
        At.svd(U_t, S_t, Vt_t);
        
        U = Vt_t.transpose();
        S = S_t.transpose();
        Vt = U_t.transpose();
        
        return;
    }

    int m = rows_;
    int n = columns_;
    
    U = this->copy();
    Matrix<T> V(n, n, "I");
    
    int maxRotations = 10 * n * n; // Heuristic
    
    for (int iter = 0; iter < maxRotations; iter++) {
        int count = 0;
        for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
                double a=0, b=0, c=0;
                for(int k=0; k<m; k++) {
                    double p = U.v[k][i];
                    double q = U.v[k][j];
                    a += p*p;
                    b += p*q;
                    c += q*q;
                }
                
                if (std::abs(b) < EPSILON) continue; // Orthogonal enough
                count++;
                
                double zeta = (c - a) / (2 * b);
                double t = (zeta > 0 ? 1 : -1) / (std::abs(zeta) + std::sqrt(1 + zeta*zeta));
                double cs = 1 / std::sqrt(1 + t*t);
                double sn = cs * t;
                
                for(int k=0; k<m; k++) {
                    double p = U.v[k][i];
                    double q = U.v[k][j];
                    U.v[k][i] = cs*p - sn*q;
                    U.v[k][j] = sn*p + cs*q;
                }
                
                for(int k=0; k<n; k++) {
                    double p = V.v[k][i];
                    double q = V.v[k][j];
                    V.v[k][i] = cs*p - sn*q;
                    V.v[k][j] = sn*p + cs*q;
                }
            }
        }
        if (count == 0) break;
    }
    
    S = Matrix<T>(n, n, 0.0); // Diagonal
    
    for(int i=0; i<n; i++) {
        double norm = 0;
        for(int k=0; k<m; k++) norm += U.v[k][i] * U.v[k][i];
        norm = std::sqrt(norm);
        S.v[i][i] = norm;
        
        if(norm > EPSILON) {
            for(int k=0; k<m; k++) U.v[k][i] /= norm;
        }
    }
    
    Vt = V.transpose();
}

template <class T>
int Matrix<T>::rank() const // Matrix Rank
{
    // Compute rank using SVD
    // Rank is the number of non-zero singular values (or > epsilon)
    Matrix<T> U, S, Vt;
    this->svd(U, S, Vt);
    
    int r = 0;
    int k = std::min(rows_, columns_);
    for(int i=0; i<k; i++) {
        double sigma = std::abs((double)S.v[i][i]);
        if (sigma > 1e-10) {
             r++;
        }
    }
    return r;
}

/**********************************************
  * Algorithm part
  **********************************************/

// Calculate the pairwise interpoint Euclidean distances
// X is data matrix, D is distance matrix
void EuclideanDistanceMatrix(Matrix<double> &X, Matrix<double> &D)
{
  int i, j, k;
  double temp;
  if (X.rows() != D.rows() || X.rows() != D.columns())
  {
    throw std::invalid_argument("Invalid distance matrix dimension.");
  }

  for (i = 0; i < D.rows(); i++)
    D.v[i][i] = 0.0;

  for (i = 0; i < D.rows() - 1; i++)
  {
    for (j = i + 1; j < D.columns(); j++)
    {
      temp = 0;
      for (k = 0; k < X.columns(); k++)
      {
        temp += std::pow(X.v[i][k] - X.v[j][k], 2);
      }
      D.v[i][j] = std::sqrt(temp);
    }
  }

  for (i = 1; i < D.rows(); i++)
  {
    for (j = 0; j < i; j++)
    {
      D.v[i][j] = D.v[j][i];
    }
  }
}

// Copy all elements of X to Y
void ElementCopy(Matrix<double> &X, Matrix<double> &Y)
{
  if (X.rows() != Y.rows() || X.columns() != Y.columns())
  {
    throw std::invalid_argument("Invalid matrix dimension.");
  }
  for (int i = 0; i < X.rows(); i++)
  {
    for (int j = 0; j < X.columns(); j++)
    {
      Y.v[i][j] = X.v[i][j];
    }
  }
}

// Multidimensional scaling (MDS)
// This function re-implements Laurens van der Maaten's MDS in his Matlab Toolbox for Dimensionality Reduction
// The Matlab MDS can be downloaded at http://crcv.ucf.edu/source/dimension
Matrix<double> MDS_UCF(Matrix<double> &D, Matrix<double> *X0, int dim, int iter)
{
  if (D.rows() != D.columns())
  {
    throw std::invalid_argument("Input distance matrix to MDS is not square.");
  }
  if (dim < 1)
  {
    throw std::invalid_argument("Invalid dimension for MDS.");
  }
  if (iter < 1)
  {
    throw std::invalid_argument("Invalid number of iterations for MDS.");
  }

  Matrix<double> X;

  // with initialization
  if (X0 != NULL)
  {
    if (X0->rows() != D.rows() || X0->columns() != dim)
    {
      throw std::invalid_argument("Input initialization to MDS has invalid dimension.");
    }
    X = X0->copy();
  }
  // without initialization
  else
  {
    X = Matrix<double>(D.rows(), dim, "rand");
    double D_mean = D.mean();                                             // mean value of distance matrix
    X.addNumberSelf(-0.5);                                                // move to the center
    X.multiplyNumberSelf(0.1 * D_mean / (1.0 / 3.0 * std::sqrt((double)dim))); // before this step, mean distance is 1/3*sqrt(d)
  }

  double lr = 0.05;  // learning rate
  double r = 2;      // metric
  int n = D.rows(); // number of vectors

  Matrix<double> dh(n, n, 0.0);
  Matrix<double> pmat(n, dim);
  Matrix<double> dhdum(n, 1);
  Matrix<double> dhmat(n - 1, dim, 0.0);

  Matrix<int> RP(n, iter, "randperm"); // the matrix for random permutation numbers
  int i, j;
  double temp;
  int m;

  printf("\n[MDS] Iterations:\n");
  for (int it = 0; it < iter; it++) // iterations
  {
    if (it % 10 == 0)
      printf("\n[MDS] ");
    printf("%3d  ", it + 1);
    for (int rp = 0; rp < n; rp++) // work on each vector in a randomly permuted order
    {
       m = RP.v[rp][it] - 1;

       for (i = 0; i < n; i++)
       {
         for (j = 0; j < dim; j++)
         {
           pmat.v[i][j] = X.v[m][j] - X.v[i][j];
         }
       }

       for (i = 0; i < n; i++)
       {
         temp = 0;
         for (j = 0; j < dim; j++)
         {
           temp += std::pow(std::fabs(pmat.v[i][j]), r);
         }
         dhdum.v[i][0] = std::pow(temp, 1.0 / r);
       }

       for (i = 0; i < n; i++)
       {
         if (i == m)
           continue;

         dh.v[m][i] = dhdum.v[i][0];
         dh.v[i][m] = dhdum.v[i][0];
       }

       for (i = 0; i < n - 1; i++)
       {
         int ii = i;
         if (i >= m)
           ii = i + 1;
         temp = lr * (dhdum.v[ii][0] - D.v[ii][m]) * std::pow(dhdum.v[ii][0], 1.0 - r);
         for (j = 0; j < dim; j++)
         {
           dhmat.v[i][j] = temp;
         }
       }

       for (i = 0; i < n - 1; i++)
       {
         int ii = i;
         if (i >= m)
           ii = i + 1;
         for (j = 0; j < dim; j++)
         {
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
// The Matlab SMACOF can be downloaded at http://tosca.cs.technion.ac.il/
Matrix<double> MDS_SMACOF(Matrix<double> &D, Matrix<double> *X0, int dim, int iter)
{
  if (D.rows() != D.columns())
  {
    throw std::invalid_argument("Input distance matrix to MDS is not square.");
  }
  if (dim < 1)
  {
    throw std::invalid_argument("Invalid dimension for MDS.");
  }
  if (iter < 1)
  {
    throw std::invalid_argument("Invalid number of iterations for MDS.");
  }

  Matrix<double> X;

  // with initialization
  if (X0 != NULL)
  {
    if (X0->rows() != D.rows() || X0->columns() != dim)
    {
      throw std::invalid_argument("Input initialization to MDS has invalid dimension.");
    }
    X = X0->copy();
  }
  // without initialization
  else
  {
    X = Matrix<double>(D.rows(), dim, "rand");
    double D_mean = D.mean();                                             // mean value of distance matrix
    X.addNumberSelf(-0.5);                                                // move to the center
    X.multiplyNumberSelf(0.1 * D_mean / (1.0 / 3.0 * std::sqrt((double)dim))); // before this step, mean distance is 1/3*sqrt(d)
  }

  Matrix<double> Z = X.copy();
  Matrix<double> D_(D.rows(), D.columns());
  Matrix<double> B(D.rows(), D.columns());
  int i, j, k;
  double temp;

  EuclideanDistanceMatrix(X, D_);

  printf("\n[MDS] Iterations:\n");
  for (int it = 0; it < iter; it++) // iterations
  {
    if (it % 10 == 0)
      printf("\n[MDS] ");
    printf("%3d  ", it + 1);

    // B = calc_B(D_,D);
    for (i = 0; i < D.rows(); i++)
    {
      for (j = 0; j < D.columns(); j++)
      {
        if (i == j || std::fabs(D_.v[i][j]) < EPSILON)
        {
          B.v[i][j] = 0.0;
        }
        else
        {
          B.v[i][j] = -D.v[i][j] / D_.v[i][j];
        }
      }
    }

    for (j = 0; j < D.columns(); j++)
    {
      temp = 0;
      for (i = 0; i < D.rows(); i++)
      {
        temp += B.v[i][j];
      }
      B.v[j][j] = -temp;
    }

    // X = B*Z/size(D,1);
    for (i = 0; i < X.rows(); i++)
    {
      for (j = 0; j < X.columns(); j++)
      {
        temp = 0;
        for (k = 0; k < B.columns(); k++)
        {
          temp += (B.v[i][k] * Z.v[k][j]);
        }
        X.v[i][j] = temp / (double)D.rows();
      }
    }

    // D_ = calc_D (X);
    EuclideanDistanceMatrix(X, D_);

    // Z = X;
    // ElementCopy(X, Z); // Wait, X and Z are Matrix objects. I can just copy!
    Z = X; // Uses move assignment or copy assignment.
  }

  printf("\n");

  return X;
}
} // namespace smat

#endif
