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
#include <cstring>
#include <ctime>
#include <cmath>
#include <initializer_list>
#include <utility>
#include <stdexcept>

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
  Matrix(int rows, int columns);                   // initialization without assigning values
  Matrix(int rows, int columns, T value);          // initialization with all same values
  Matrix(int rows, int columns, std::string type); // special matrix such as I
  Matrix(const char *filename);                    // load matrix from txt file
  Matrix(const Matrix &other);                     // copy constructor
  Matrix(Matrix &&other);                          // move constructor
  Matrix(std::initializer_list<std::initializer_list<T>> list); // initializer list
  Matrix &operator=(const Matrix &other);          // assignment operator
  Matrix &operator=(Matrix &&other);               // move assignment operator
  ~Matrix();                                       // destruction

  void set(int r, int c, T value); // row, column, value
  T get(int r, int c);             // row, column
  int rows();                      // number of rows
  int columns();                   // number of columns

  void print();   // print the matrix
  Matrix *copy(); // copy itself to a new matrix

  void saveTxt(const char *filename); // save matrix to txt file

  // B=M'
  Matrix *transpose();
  // B=M(r1:r2,c1:c2)
  Matrix *sub(int r1, int r2, int c1, int c2); // submatrix
  // B=|M|
  Matrix *abs(); // absolute values

  // numbers of matrix
  T trace();               // trace
  double fnorm();          // Frobenius norm
  double pnorm(double p);  // p-norm
  T maxEl(int &r, int &c); // max element
  T minEl(int &r, int &c); // min element
  double mean();           // mean of elements
  T sum();                 // sum of elements
  double std();            // standard deviation of elements

  // M=M+a
  void addNumberSelf(T value); // add a number to itself in space
  // M=M-a
  void subtractNumberSelf(T value); // subtract a number from itself in space
  // M=M*a
  void multiplyNumberSelf(T value); // multiply a number to itself in space
  // M=M/a
  void divideNumberSelf(T value); // divide itself by a number in space

  // Element-wise functions
  Matrix *exp();
  Matrix *log();
  Matrix *sqrt();
  Matrix *power(double p);

  // M=M+A
  void addMatrixSelf(Matrix *A); // add a matrix to itself in space
  // M=M-A
  void subtractMatrixSelf(Matrix *A); // subtract a matrix from itself in space
  // M=M.*A
  void dotMultiplyMatrixSelf(Matrix *A); // dot multiply a matrix to itself in space
  // M=M./A
  void dotDivideMatrixSelf(Matrix *A); // dot divide a matrix to itself in space

  // B=M+A
  Matrix *addMatrixNew(Matrix *A); // add a matrix to itself with new matrix
  // B=M-A
  Matrix *subtractMatrixNew(Matrix *A); // subtract a matrix from itself with new matrix
  // B=M.*A
  Matrix *dotMultiplyMatrixNew(Matrix *A); // dot multiply a matrix to itself with new matrix
  // B=M./A
  Matrix *dotDivideMatrixNew(Matrix *A); // dot divide a matrix to itself with new matrix
  // B=M*A
  Matrix *multiplyMatrixNew(Matrix *A); // multiply a matrix to itself with new matrix

  // Concatenation
  Matrix *concatenateRight(Matrix *A);
  Matrix *concatenateBottom(Matrix *A);

  // Linear Algebra
  void lu(Matrix *&L, Matrix *&U, Matrix *&P);   // LU Decomposition
  void qr(Matrix *&Q, Matrix *&R);               // QR Decomposition
  void eigen(Matrix *&V, Matrix *&D);            // Eigenvalue Decomposition
  void svd(Matrix *&U, Matrix *&S, Matrix *&Vt); // Singular Value Decomposition
  T determinant();                               // Determinant
  int rank();                                    // Matrix Rank
  Matrix *inverse();                             // Matrix Inverse
  Matrix *solve(Matrix *B);                      // Solve Ax=B

  // Operator overloading
  Matrix<T> operator+(const Matrix<T> &other);
  Matrix<T> operator-(const Matrix<T> &other);
  Matrix<T> operator*(const Matrix<T> &other);

private:
  int rows_;
  int columns_;
  T **v;
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
Matrix<T>::Matrix(int rows, int columns) // initialization without assigning values
{
  if (rows < 1 || columns < 1)
  {
    throw std::invalid_argument("Invalid construction arguments: rows=" + std::to_string(rows) + ", columns=" + std::to_string(columns));
  }

  rows_ = rows;
  columns_ = columns;

  v = new T *[rows];
  v[0] = new T[rows * columns];
  for (int i = 1; i < rows; i++)
  {
    v[i] = v[i - 1] + columns;
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

  v = new T *[rows];
  v[0] = new T[rows * columns];
  for (int i = 1; i < rows; i++)
  {
    v[i] = v[i - 1] + columns;
  }
  
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      v[i][j] = value;
    }
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

  v = new T *[rows];
  v[0] = new T[rows * columns];
  for (int i = 1; i < rows; i++)
  {
    v[i] = v[i - 1] + columns;
  }

  if (type.compare("I") == 0)
  {
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < columns; j++)
      {
        if (i == j)
          v[i][j] = (T)1;
        else
          v[i][j] = (T)0;
      }
    }
  }

  else if (type.compare("rand") == 0) // all elements between 0 and 1
  {
    srand(time(NULL));
    int r1;
    double r2;
    for (int i = 0; i < rows_; i++)
    {
      for (int j = 0; j < columns_; j++)
      {
        r1 = rand() * rand() + rand() * rand() + rand();
        if (r1 < 0)
          r1 = -r1;
        r2 = double(r1 % 1000001) / 1000000;

        v[i][j] = (T)r2;
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
    for (int j = 0; j < columns; j++)
    {
      for (int i = 0; i < rows; i++)
      {
        v[i][j] = i + 1;
      }

      for (int i = 0; i < rows; i++)
      {
        int k = rand() % rows;
        if (k >= rows || k < 0)
        {
          printf("Invalid row index: %d\n", k);
          exit(1);
        }
        T temp = v[i][j];
        v[i][j] = v[k][j];
        v[k][j] = temp;
      }
    }
  }

  else
  {
    throw std::invalid_argument("Undefined matrix type: " + type);
  }
}

template <class T>
Matrix<T>::Matrix(Matrix &&other) : rows_(0), columns_(0), v(NULL) // move constructor
{
  rows_ = other.rows_;
  columns_ = other.columns_;
  v = other.v;

  other.rows_ = 0;
  other.columns_ = 0;
  other.v = NULL;
}

template <class T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list) // initializer list
{
  rows_ = list.size();
  if (rows_ == 0)
  {
    columns_ = 0;
    v = NULL;
    return;
  }
  columns_ = list.begin()->size();

  v = new T *[rows_];
  v[0] = new T[rows_ * columns_];
  for(int i=1; i<rows_; i++) {
        v[i] = v[i-1] + columns_;
  }
  
  int i = 0;
  for (const auto &row : list)
  {
    if ((int)row.size() != columns_)
    {
      printf("All rows in initializer list must have the same size.\n");
      exit(1);
    }
    // v[i] is already set
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
Matrix<T> &Matrix<T>::operator=(Matrix &&other) // move assignment operator
{
  if (this != &other)
  {
    // release current memory
    if (v != NULL)
    {
      delete[] v[0];
      delete[] v;
    }

    // steal resources
    rows_ = other.rows_;
    columns_ = other.columns_;
    v = other.v;

    // nullify other
    other.rows_ = 0;
    other.columns_ = 0;
    other.v = NULL;
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
  v = new T *[rows];
  v[0] = new T[rows * columns];
  for (int i = 1; i < rows; i++)
  {
    v[i] = v[i - 1] + columns;
  }


  pFile = fopen(filename, "r");
  if (pFile == NULL)
  {
    printf("File \"%s\" cannot be found.\n", filename);
    exit(1);
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
  v = new T *[rows_];
  v[0] = new T[rows_ * columns_];
  for (int i = 1; i < rows_; i++)
  {
    v[i] = v[i - 1] + columns_;
  }

  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      v[i][j] = other.v[i][j];
    }
  }
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const Matrix &other) // assignment operator
{
  if (this != &other)
  {
    if (rows_ != other.rows_ || columns_ != other.columns_)
    {
      if (v != NULL)
      {
        delete[] v[0];
        delete[] v;
      }
      rows_ = other.rows_;
      columns_ = other.columns_;
      v = new T *[rows_];
      v[0] = new T[rows_ * columns_];
      for (int i = 1; i < rows_; i++)
      {
        v[i] = v[i - 1] + columns_;
      }
    }
    for (int i = 0; i < rows_; i++)
    {
      for (int j = 0; j < columns_; j++)
      {
        v[i][j] = other.v[i][j];
      }
    }
  }
  return *this;
}

template <class T>
Matrix<T>::~Matrix() // destruction
{
  if (v != NULL)
  {
    delete[] v[0]; // delete data
    delete[] v;    // delete row pointers
  }
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
T Matrix<T>::get(int r, int c) // row, column
{
  if (r < 0 || r >= rows_ || c < 0 || c >= columns_)
  {
    throw std::out_of_range("Invalid index in get(): r=" + std::to_string(r) + ", c=" + std::to_string(c));
  }
  return v[r][c];
}

template <class T>
int Matrix<T>::rows() // number of rows
{
  return rows_;
}

template <class T>
int Matrix<T>::columns() // number of columns
{
  return columns_;
}

template <class T>
void Matrix<T>::print() // print the matrix
{
  printf("\n");
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      printf("%8.3f    ", (double)v[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

template <class T>
Matrix<T> *Matrix<T>::copy() // copy itself to a new matrix
{
  Matrix<T> *A = new Matrix<T>(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      A->set(i, j, v[i][j]);
    }
  }
  return A;
}

template <class T>
void Matrix<T>::saveTxt(const char *filename)
{
  FILE *pFile;
  pFile = fopen(filename, "w");
  if (pFile == NULL)
  {
    throw std::runtime_error(std::string("Cannot save to file \"") + filename + "\".");
  }
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      fprintf(pFile, "%f  ", (double)v[i][j]);
    }
    fprintf(pFile, "\n");
  }
  fclose(pFile);
  printf("Matrix saved to file \"%s\"\n", filename);
}

template <class T>
Matrix<T> *Matrix<T>::transpose()
{
  Matrix<T> *A = new Matrix<T>(columns_, rows_);
  for (int i = 0; i < columns_; i++)
  {
    for (int j = 0; j < rows_; j++)
    {
      A->set(i, j, v[j][i]);
    }
  }
  return A;
}

template <class T>
Matrix<T> *Matrix<T>::sub(int r1, int r2, int c1, int c2) // submatrix
{
  if (r1 < 0 || r1 >= rows_ || r2 < 0 || r2 >= rows_ || r2 < r1 || c1 < 0 || c1 >= columns_ || c2 < 0 || c2 >= columns_ || c2 < c1)
  {
    throw std::out_of_range("Invalid submatrix indices.");
  }

  int newRows = r2 - r1 + 1;
  int newColumns = c2 - c1 + 1;
  Matrix<T> *A = new Matrix<T>(newRows, newColumns);
  for (int i = 0; i < newRows; i++)
  {
    for (int j = 0; j < newColumns; j++)
    {
      A->set(i, j, v[i + r1][j + c1]);
    }
  }
  return A;
}

template <class T>
Matrix<T> *Matrix<T>::abs() // absolute values
{
  Matrix<T> *A = new Matrix<T>(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      A->set(i, j, v[i][j] > 0 ? v[i][j] : -v[i][j]);
    }
  }
  return A;
}

template <class T>
T Matrix<T>::trace() // trace
{
  T x = 0;
  for (int i = 0; i < min<int>(rows_, columns_); i++)
  {
    x += v[i][i];
  }
  return x;
}

template <class T>
double Matrix<T>::fnorm() // Frobenius norm
{
  double x = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      x += (v[i][j] * v[i][j]);
    }
  }
  return std::sqrt(x);
}

template <class T>
double Matrix<T>::pnorm(double p) // p-norm
{
  double x = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      x += pow(fabs((double)v[i][j]), p);
    }
  }
  if (p == 0.0)
  {
    return x;
  }
  return pow(x, 1 / p);
}

template <class T>
T Matrix<T>::maxEl(int &r, int &c) // max element
{
  T x = v[0][0];
  r = 0;
  c = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      if (v[i][j] > x)
      {
        x = v[i][j];
        r = i;
        c = j;
      }
    }
  }
  return x;
}

template <class T>
T Matrix<T>::minEl(int &r, int &c) // min element
{
  T x = v[0][0];
  r = 0;
  c = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      if (v[i][j] < x)
      {
        x = v[i][j];
        r = i;
        c = j;
      }
    }
  }
  return x;
}

template <class T>
double Matrix<T>::mean() // mean of elements
{
  double x = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      x += (double)v[i][j];
    }
  }
  return x / rows_ / columns_;
}

template <class T>
T Matrix<T>::sum() // sum of elements
{
  T x = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      x += v[i][j];
    }
  }
  return x;
}

template <class T>
double Matrix<T>::std() // standard deviation of elements
{
  double m = mean();
  double s = 0;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      s += (v[i][j] - m) * (v[i][j] - m);
    }
  }
  s = s / rows_ / columns_;
  return sqrt(s);
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
Matrix<T> *Matrix<T>::exp()
{
  Matrix<T> *A = new Matrix<T>(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      A->set(i, j, std::exp((double)v[i][j]));
    }
  }
  return A;
}

template <class T>
Matrix<T> *Matrix<T>::log()
{
  Matrix<T> *A = new Matrix<T>(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      A->set(i, j, std::log((double)v[i][j]));
    }
  }
  return A;
}

template <class T>
Matrix<T> *Matrix<T>::sqrt()
{
  Matrix<T> *A = new Matrix<T>(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      A->set(i, j, std::sqrt((double)v[i][j]));
    }
  }
  return A;
}

template <class T>
Matrix<T> *Matrix<T>::power(double p)
{
  Matrix<T> *A = new Matrix<T>(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      A->set(i, j, std::pow((double)v[i][j], p));
    }
  }
  return A;
}

template <class T>
void Matrix<T>::addMatrixSelf(Matrix *A) // add a matrix to itself in space
{
  if (rows_ != A->rows() || columns_ != A->columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix summation.");
  }

  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      v[i][j] += A->get(i, j);
    }
  }
}

template <class T>
void Matrix<T>::subtractMatrixSelf(Matrix *A)
{
  if (rows_ != A->rows() || columns_ != A->columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix subtraction.");
  }

  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      v[i][j] -= A->get(i, j);
    }
  }
}

template <class T>
void Matrix<T>::dotMultiplyMatrixSelf(Matrix *A) // dot multiply a matrix to itself in space
{
  if (rows_ != A->rows() || columns_ != A->columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix dot multiplication.");
  }

  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      v[i][j] *= A->get(i, j);
    }
  }
}

template <class T>
void Matrix<T>::dotDivideMatrixSelf(Matrix *A)
{
  if (rows_ != A->rows() || columns_ != A->columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix dot division.");
  }

  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      v[i][j] /= A->get(i, j);
    }
  }
}

template <class T>
Matrix<T> *Matrix<T>::addMatrixNew(Matrix *A) // add a matrix to itself with new matrix
{
  if (rows_ != A->rows() || columns_ != A->columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix summation.");
  }

  Matrix<T> *B = new Matrix<T>(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B->set(i, j, v[i][j] + A->get(i, j));
    }
  }
  return B;
}

template <class T>
Matrix<T> *Matrix<T>::subtractMatrixNew(Matrix *A)
{
  if (rows_ != A->rows() || columns_ != A->columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix subtraction.");
  }

  Matrix<T> *B = new Matrix<T>(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B->set(i, j, v[i][j] - A->get(i, j));
    }
  }
  return B;
}

template <class T>
Matrix<T> *Matrix<T>::dotMultiplyMatrixNew(Matrix *A) // dot multiply a matrix to itself with new matrix
{
  if (rows_ != A->rows() || columns_ != A->columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix dot multiplication.");
  }

  Matrix<T> *B = new Matrix<T>(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B->set(i, j, v[i][j] * A->get(i, j));
    }
  }
  return B;
}

template <class T>
Matrix<T> *Matrix<T>::dotDivideMatrixNew(Matrix *A)
{
  if (rows_ != A->rows() || columns_ != A->columns())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix dot division.");
  }

  Matrix<T> *B = new Matrix<T>(rows_, columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < columns_; j++)
    {
      B->set(i, j, v[i][j] / A->get(i, j));
    }
  }
  return B;
}

template <class T>
Matrix<T> *Matrix<T>::multiplyMatrixNew(Matrix *A) // multiply a matrix to itself with new matrix
{
  if (columns_ != A->rows())
  {
    throw std::invalid_argument("Unmatched matrix sizes in matrix multiplication.");
  }

  Matrix<T> *B = new Matrix<T>(rows_, A->columns());
  T temp;
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < A->columns(); j++)
    {
      temp = 0;
      for (int k = 0; k < columns_; k++)
      {
        temp += (v[i][k] * A->get(k, j));
      }
      B->set(i, j, temp);
    }
  }
  return B;
}

template <class T>
Matrix<T> *Matrix<T>::concatenateRight(Matrix *A)
{
  if (rows_ != A->rows())
  {
    throw std::invalid_argument("Unmatched matrix rows in concatenation.");
  }
  Matrix<T> *B = new Matrix<T>(rows_, columns_ + A->columns());
  for (int i = 0; i < rows_; i++)
  {
    for(int j=0; j<columns_; j++) B->set(i, j, v[i][j]);
    for(int j=0; j<A->columns(); j++) B->set(i, j + columns_, A->get(i, j));
  }
  return B;
}

template <class T>
Matrix<T> *Matrix<T>::concatenateBottom(Matrix *A)
{
  if (columns_ != A->columns())
  {
    throw std::invalid_argument("Unmatched matrix columns in concatenation.");
  }
  Matrix<T> *B = new Matrix<T>(rows_ + A->rows(), columns_);
  for(int j=0; j<columns_; j++)
  {
    for(int i=0; i<rows_; i++) B->set(i, j, v[i][j]);
    for(int i=0; i<A->rows(); i++) B->set(i + rows_, j, A->get(i, j));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other)
{
  if (rows_ != other.rows_ || columns_ != other.columns_)
  {
    throw std::invalid_argument("Unmatched matrix sizes in operator+.");
  }
  Matrix<T> res(rows_, columns_);
  for(int i=0; i<rows_; i++)
    for(int j=0; j<columns_; j++)
      res.set(i, j, v[i][j] + other.v[i][j]);
  return res;
}

template <class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &other)
{
  if (rows_ != other.rows_ || columns_ != other.columns_)
  {
    throw std::invalid_argument("Unmatched matrix sizes in operator-.");
  }
  Matrix<T> res(rows_, columns_);
  for(int i=0; i<rows_; i++)
    for(int j=0; j<columns_; j++)
      res.set(i, j, v[i][j] - other.v[i][j]);
  return res;
}

template <class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &other)
{
  if (columns_ != other.rows_)
  {
    throw std::invalid_argument("Unmatched matrix sizes in operator*.");
  }
  Matrix<T> res(rows_, other.columns_);
  for (int i = 0; i < rows_; i++)
  {
    for (int j = 0; j < other.columns_; j++)
    {
      T temp = 0;
      for (int k = 0; k < columns_; k++)
      {
        temp += (v[i][k] * other.v[k][j]);
      }
      res.set(i, j, temp);
    }
  }
  return res;
}

template <class T>
void Matrix<T>::lu(Matrix *&L, Matrix *&U, Matrix *&P) // LU Decomposition
{
  if (rows_ != columns_)
  {
    throw std::invalid_argument("LU decomposition requires square matrix.");
  }
  int n = rows_;
  L = new Matrix<T>(n, n, 0.0);
  U = new Matrix<T>(n, n, 0.0);
  P = new Matrix<T>(n, n, "I");

  Matrix<T> *A = this->copy(); // Work on a copy

  for (int i = 0; i < n; i++)
  {
    // Pivot
    T maxEl = std::abs((double)A->get(i, i));
    int maxRow = i;
    for (int k = i + 1; k < n; k++)
    {
      if (std::abs((double)A->get(k, i)) > maxEl)
      {
        maxEl = std::abs((double)A->get(k, i));
        maxRow = k;
      }
    }

    // Swap rows in A and P
    for (int k = 0; k < n; k++)
    {
      T tmp = A->get(i, k);
      A->set(i, k, A->get(maxRow, k));
      A->set(maxRow, k, tmp);

      tmp = P->get(i, k);
      P->set(i, k, P->get(maxRow, k));
      P->set(maxRow, k, tmp);
      
      // We also need to swap rows in L (the parts already computed)?
      // Actually standard algorithm: P A = L U. L is lower triangular.
      // If we swap rows of A, we are essentially permuting equations.
      // Standard pivot strategy involves keeping track of permutations.
      // Doolittle algorithm extracts L and U directly.
    }
    
    // However, simplest standard implementation with P A = L U usually does:
    // 1. Pivot A. Record swap in P.
    // 2. But we must also swap rows in L for k < i to keep L lower triangular consistent!
    for (int k=0; k<i; k++) {
        T tmp = L->get(i, k);
        L->set(i, k, L->get(maxRow, k));
        L->set(maxRow, k, tmp);
    }
    
    L->set(i, i, 1.0);

    for (int k = i + 1; k < n; k++)
    {
      double c = -A->get(k, i) / A->get(i, i);
      L->set(k, i, -c); // Multiplier
      for (int j = i; j < n; j++)
      {
        if (i == j)
          A->set(k, j, 0);
        else
          A->set(k, j, A->get(k, j) + c * A->get(i, j));
      }
    }
  }

  // U is the remaining A
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      if (j >= i)
        U->set(i, j, A->get(i, j));
    }
  }

  delete A;
}

template <class T>
T Matrix<T>::determinant() // Determinant
{
  if (rows_ != columns_)
  {
    throw std::invalid_argument("Determinant requires square matrix.");
  }
  
  // Custom simple Gaussian elimination for det to avoid full LU overhead if desired,
  // but reusing LU code is better for maintenance unless perf is critical.
  // Actually, I'll implement a tailored Gaussian elimination for Det to track sign flips easily.
  
  Matrix<T> *A = this->copy();
  T det = 1.0;
  int n = rows_;
  
  for (int i = 0; i < n; i++)
  {
    int pivot = i;
    for (int j = i + 1; j < n; j++) {
      if (std::abs((double)A->get(j, i)) > std::abs((double)A->get(pivot, i)))
        pivot = j; 
    }
    
    if (std::abs((double)A->get(pivot, i)) < EPSILON) {
      delete A;
      return 0.0;
    }
    
    if (pivot != i) {
       // Swap rows
       for(int k=0; k<n; k++) {
         T tmp = A->get(i, k);
         A->set(i, k, A->get(pivot, k));
         A->set(pivot, k, tmp);
       }
       det = -det;
    }
    
    det *= A->get(i, i);
    
    for (int j = i + 1; j < n; j++) {
      double factor = A->get(j, i) / A->get(i, i);
      for (int k = i + 1; k < n; k++) {
         A->set(j, k, A->get(j, k) - factor * A->get(i, k));
      }
    }
  }
  
  delete A;
  return det;
}

template <class T>
Matrix<T>* Matrix<T>::inverse() // Matrix Inverse
{
  if (rows_ != columns_)
  {
    printf("Inverse requires square matrix.\n");
    exit(1);
  }
  
  int n = rows_;
  Matrix<T> *Inv = new Matrix<T>(n, n);
  Matrix<T> *A = this->copy();
  // Augment with Identity? Or just solve Ax = I column by column.
  // Gaussian elimination with Identity.
  
  // Initialize Inv as Identity
  for(int i=0; i<n; i++)
    for(int j=0; j<n; j++)
      Inv->set(i, j, (i==j) ? 1.0 : 0.0);
      
  for (int i = 0; i < n; i++) {
      int pivot = i;
      for(int j=i+1; j<n; j++)
        if(std::abs((double)A->get(j, i)) > std::abs((double)A->get(pivot, i))) pivot = j;
        
      if(std::abs((double)A->get(pivot, i)) < EPSILON) {
          throw std::runtime_error("Matrix is singular, cannot find inverse.");
      }
      
      // Swap rows in A and Inv
      if(pivot != i) {
          for(int k=0; k<n; k++) {
              T tmp = A->get(i, k); A->set(i, k, A->get(pivot, k)); A->set(pivot, k, tmp);
              tmp = Inv->get(i, k); Inv->set(i, k, Inv->get(pivot, k)); Inv->set(pivot, k, tmp);
          }
      }
      
      // Scale row i to make diagonal 1
      double div = A->get(i, i);
      for(int k=0; k<n; k++) {
          A->set(i, k, A->get(i, k) / div);
          Inv->set(i, k, Inv->get(i, k) / div);
      }
      
      // Eliminate other rows
      for(int j=0; j<n; j++) {
          if(i != j) {
              double mul = A->get(j, i);
              for(int k=0; k<n; k++) {
                  A->set(j, k, A->get(j, k) - mul * A->get(i, k));
                  Inv->set(j, k, Inv->get(j, k) - mul * Inv->get(i, k));
              }
          }
      }
  }
  
  delete A;
  return Inv;
}

template <class T>
Matrix<T>* Matrix<T>::solve(Matrix *B) // Solve Ax=B
{
    if (rows_ != columns_)
    {
        throw std::invalid_argument("Matrix must be square to solve system.");
    }
    if (rows_ != B->rows())
    {
        throw std::invalid_argument("Matrix row dimensions must agree.");
    }

    Matrix<T> *L, *U, *P;
    this->lu(L, U, P);

    // Ax = B -> LUx = PB
    // Let y = Ux, then Ly = PB
    
    // 1. Compute PB
    Matrix<T> *PB = P->multiplyMatrixNew(B);
    
    // 2. Solve Ly = PB (Forward substitution)
    Matrix<T> *y = new Matrix<T>(rows_, B->columns());
    for(int k=0; k<B->columns(); k++)
    {
        for(int i=0; i<rows_; i++)
        {
            double sum = 0;
            for(int j=0; j<i; j++)
            {
                sum += (double)L->get(i, j) * y->get(j, k);
            }
            y->set(i, k, (PB->get(i, k) - sum)); // L diagonal is 1
        }
    }
    
    // 3. Solve Ux = y (Backward substitution)
    Matrix<T> *x = new Matrix<T>(rows_, B->columns());
    for(int k=0; k<B->columns(); k++)
    {
        for(int i=rows_-1; i>=0; i--)
        {
            double sum = 0;
            for(int j=i+1; j<columns_; j++)
            {
                sum += (double)U->get(i, j) * x->get(j, k);
            }
            if (std::abs((double)U->get(i, i)) < EPSILON)
            {
                throw std::runtime_error("Matrix is singular.");
            }
            x->set(i, k, (y->get(i, k) - sum) / (double)U->get(i, i));
        }
    }
    
    delete L; delete U; delete P; delete PB; delete y;
    return x;
}

template <class T>
void Matrix<T>::qr(Matrix *&Q, Matrix *&R) // QR Decomposition
{
    // Householder transformations
    // A = Q R
    // R is upper triangular, Q is orthogonal
    if (rows_ < columns_) {
        throw std::invalid_argument("QR decomposition requires rows >= columns.");
    }
    
    int m = rows_;
    int n = columns_;
    
    Q = new Matrix<T>(m, m, "I");
    R = this->copy();
    
    for (int k = 0; k < std::min(m-1, n); k++) {
        // Construct vector x = R[k:m, k]
        // We want to zero out elements below diagonal in column k
        
        // Compute norm of x
        double norm_x = 0;
        for(int i=k; i<m; i++) {
            norm_x += R->get(i, k) * R->get(i, k);
        }
        norm_x = std::sqrt(norm_x);
        
        // Check if column is already zero (or close)
        if (norm_x < EPSILON) continue;
        
        // v = x + sign(x[0]) * ||x|| * e1
        // Actually, better numerical stability: v[0] = x[0] + sign(x[0])*norm_x
        // We only store the non-zero part of v, which is size (m-k)
        
        double u1 = R->get(k, k);
        double sign = (u1 >= 0) ? 1.0 : -1.0;
        double u1_new = u1 + sign * norm_x;
        
        // v = [u1_new, x_1, x_2, ...]
        // normalize v? 
        // H = I - 2 v v^T / (v^T v)
        
        double norm_v_sq = u1_new * u1_new; // first element squared
        for(int i=k+1; i<m; i++) {
            norm_v_sq += R->get(i, k) * R->get(i, k);
        }
        
        if (norm_v_sq < EPSILON) continue;
        
        double tau = 2.0 / norm_v_sq;
        
        // Apply H to R: R = (I - tau v v^T) R = R - tau v (v^T R)
        // Only affects rows k to m
        
        // Compute w = v^T R (row vector)
        // v is column vector of size m-k.
        // R subblock is (m-k) x (n-k)
        
        for (int j = k; j < n; j++) {
            double dot = u1_new * R->get(k, j);
            for(int i=k+1; i<m; i++) {
                dot += R->get(i, k) * R->get(i, j); 
                // Wait, constructing v explicitly means v[i-k] = R[i, k] for i>k
            }
            
            // R[k:m, j] -= tau * dot * v
            double val = R->get(k, j) - tau * dot * u1_new;
            R->set(k, j, val);
            for(int i=k+1; i<m; i++) {
                // v[i-k] corresponds to old R[i, k]
                // But we are modifying R in place!
                // We need to use "v" which was derived from OLD column k.
                // Ah, column k of R for index i>k IS v[i-k]!
                // So for j=k?
                // For j=k: dot...
                // Wait, if we modify column k, we lose v.
                // We must store v or update column k last.
                // Or better, just store v separately.
                
            }
        }
    }
    // Let's rewrite cleanly with explicit v vector.
    delete Q; delete R; // cleanup retry
    
    Q = new Matrix<T>(m, m, "I");
    R = this->copy();
    
    double* v = new double[m];
    
    for (int k = 0; k < std::min(m-1, n); k++) {
        double norm_x = 0;
        for(int i=k; i<m; i++) norm_x += R->get(i, k) * R->get(i, k);
        norm_x = std::sqrt(norm_x);
        
        if(norm_x < EPSILON) continue;
        
        double u1 = R->get(k, k);
        double sign = (u1 >= 0) ? 1.0 : -1.0;
        
        // Construct v
        v[k] = u1 + sign * norm_x;
        for(int i=k+1; i<m; i++) v[i] = R->get(i, k);
        
        double norm_v_sq = 0;
        for(int i=k; i<m; i++) norm_v_sq += v[i]*v[i];
        
        if (norm_v_sq < EPSILON) continue;
        double tau = 2.0 / norm_v_sq;
        
        // Update R: R = H R
        // w = v^T R
        for (int j = k; j < n; j++) {
            double dot = 0;
            for(int i=k; i<m; i++) dot += v[i] * R->get(i, j);
             
            for(int i=k; i<m; i++) {
                R->set(i, j, R->get(i, j) - tau * dot * v[i]);
            }
        }
        
        // Update Q: Q = Q H (accumulate from right because H_1 H_2 ... )
        // Q_new = Q_old * (I - tau v v^T) = Q_old - tau (Q_old v) v^T
        for (int i=0; i<m; i++) { // For each row of Q
             double dot = 0;
             for(int j=k; j<m; j++) dot += Q->get(i, j) * v[j];
             
             for(int j=k; j<m; j++) {
                 Q->set(i, j, Q->get(i, j) - tau * dot * v[j]);
             }
        }
    }
    delete[] v;
}

template <class T>
void Matrix<T>::eigen(Matrix *&V, Matrix *&D) // Eigenvalue Decomposition
{
    // Jacobi eigenvalue algorithm to find eigenvalues and eigenvectors of a real symmetric matrix
    // A V = V D
    if (rows_ != columns_)
    {
      throw std::invalid_argument("Eigen decomposition requires square matrix.");
    }
    
    int n = rows_;
    D = this->copy(); // Will become diagonal
    V = new Matrix<T>(n, n, "I"); // Accumulate rotations
    
    int maxRotations = 5 * n * n;
    
    for (int i = 0; i < maxRotations; i++) {
        // Find largest off-diagonal element
        double maxEl = 0;
        int p = -1, q = -1;
        
        for (int r = 0; r < n-1; r++) {
            for (int c = r+1; c < n; c++) {
                if (std::abs((double)D->get(r,c)) > maxEl) {
                    maxEl = std::abs((double)D->get(r,c));
                    p = r; q = c;
                }
            }
        }
        
        if (maxEl < EPSILON) break; // Converged
        
        double App = D->get(p, p);
        double Aqq = D->get(q, q);
        double Apq = D->get(p, q);
        
        double theta = 0.5 * std::atan2(2 * Apq, Aqq - App);
        double c = std::cos(theta);
        double s = std::sin(theta);
        
        // Update D (A' = J^T A J)
        // D[p,p] = c^2 App - 2sc Apq + s^2 Aqq
        // D[q,q] = s^2 App + 2sc Apq + c^2 Aqq
        // D[p,q] = D[q,p] = 0
        // Other elements:
        // D[p,k] = c D[p,k] - s D[q,k]
        // D[q,k] = s D[p,k] + c D[q,k]
        
        D->set(p, p, c*c*App - 2*s*c*Apq + s*s*Aqq);
        D->set(q, q, s*s*App + 2*s*c*Apq + c*c*Aqq);
        D->set(p, q, 0.0);
        D->set(q, p, 0.0);
        
        for (int k=0; k<n; k++) {
            if (k!=p && k!=q) {
                double Dpk = D->get(p, k);
                double Dqk = D->get(q, k);
                D->set(p, k, c*Dpk - s*Dqk);
                D->set(k, p, D->get(p, k)); // symmetry
                
                D->set(q, k, s*Dpk + c*Dqk);
                D->set(k, q, D->get(q, k));
            }
        }
        
        // Update V (V' = V J)
        // V[k,p] = c V[k,p] - s V[k,q]
        // V[k,q] = s V[k,p] + c V[k,q]
        for (int k=0; k<n; k++) {
            double Vkp = V->get(k, p);
            double Vkq = V->get(k, q);
            V->set(k, p, c*Vkp - s*Vkq);
            V->set(k, q, s*Vkp + c*Vkq);
        }
    }
}



template <class T>
void Matrix<T>::svd(Matrix *&U, Matrix *&S, Matrix *&Vt) // Singular Value Decomposition
{
    // One-sided Jacobi SVD
    // A = U S Vt
    
    // Handle m < n by recursion on A^T
    if (rows_ < columns_) {
        Matrix<T> *At = this->transpose();
        Matrix<T> *U_t, *S_t, *Vt_t;
        At->svd(U_t, S_t, Vt_t);
        
        // A = (U S Vt)^T = V S^T U^T = V S U^T
        // So U_new = V_old, S_new = S_old^T = S_old, Vt_new = U_old^T
        
        U = Vt_t->transpose(); // Wait, Vt_t is V^T. Transpose gives V.
        S = S_t->transpose(); // S is diagonal usually, but dimensions flip? 
        // Our S will be diagonal square or match dimensions?
        // Ideally S is diagonal. Let's make S diagonal (min(m,n) size or m x n dimensions?).
        // Standard: U is m x m, S is m x n, Vt is n x n. Or thin SVD.
        // Let's implement Thin SVD or standard full SVD?
        // One-Sided Jacobi naturally gives Thin SVD (U is m x n, S is n x n, V is n x n) if m >= n.
        
        // Let's stick to simple convention: U (m x min), S (diagonal), Vt (min x n)?
        // No, standard full SVD: U (m x m), S (m x n), Vt (n x n).
        // For simplicity allow Thin SVD logic where U corresponds to non-zero singular values columns.
        
        // Actually, let's look at the recursion again.
        // At (n x m) -> U' (n x n), S' (n x m), Vt' (m x m).
        // A = (U' S' Vt')^T = Vt'^T S'^T U'^T
        // So U = Vt'^T, S = S'^T, Vt = U'^T
        
        U = Vt_t->transpose();
        S = S_t->transpose();
        Vt = U_t->transpose(); // U' is n x n, so Vt is n x n.
        
        delete At; delete U_t; delete S_t; delete Vt_t;
        return;
    }

    int m = rows_;
    int n = columns_;
    
    // Copy A into U initially (we will orthogonalize columns of U)
    U = this->copy();
    // V starts as Identity
    Matrix<T> *V = new Matrix<T>(n, n, "I");
    
    // One-sided Jacobi
    int maxRotations = 10 * n * n; // Heuristic
    
    for (int iter = 0; iter < maxRotations; iter++) {
        int count = 0;
        for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
                // Compute [a b; b c] of column i and j
                // a = col_i . col_i
                // b = col_i . col_j
                // c = col_j . col_j
                
                double a=0, b=0, c=0;
                for(int k=0; k<m; k++) {
                    double p = U->get(k, i);
                    double q = U->get(k, j);
                    a += p*p;
                    b += p*q;
                    c += q*q;
                }
                
                if (std::abs(b) < EPSILON) continue; // Orthogonal enough
                count++;
                
                // Jacobi rotation
                double zeta = (c - a) / (2 * b);
                double t = (zeta > 0 ? 1 : -1) / (std::abs(zeta) + std::sqrt(1 + zeta*zeta));
                double cs = 1 / std::sqrt(1 + t*t);
                double sn = cs * t;
                
                // Update U columns i and j
                for(int k=0; k<m; k++) {
                    double p = U->get(k, i);
                    double q = U->get(k, j);
                    U->set(k, i, cs*p - sn*q);
                    U->set(k, j, sn*p + cs*q);
                }
                
                // Update V columns i and j
                // V_new = V * J(i, j, theta). 
                // Col i of V = c * col i - s * col j
                for(int k=0; k<n; k++) {
                    double p = V->get(k, i);
                    double q = V->get(k, j);
                    V->set(k, i, cs*p - sn*q);
                    V->set(k, j, sn*p + cs*q);
                }
            }
        }
        if (count == 0) break;
    }
    
    // Norms of columns of U are singular values
    S = new Matrix<T>(n, n, 0.0); // Diagonal
    // U currently contains A V. 
    // Norms are singular values. 
    
    for(int i=0; i<n; i++) {
        double norm = 0;
        for(int k=0; k<m; k++) norm += U->get(k, i) * U->get(k, i);
        norm = std::sqrt(norm);
        S->set(i, i, norm);
        
        // Normalize column i of U
        if (norm > EPSILON) {
            for(int k=0; k<m; k++) U->set(k, i, U->get(k, i) / norm);
        }
    }
    
    // U is currently m x n (Compact SVD)
    // S is n x n
    // Vt is V^T
    Vt = V->transpose();
    delete V;
}

template <class T>
int Matrix<T>::rank() // Matrix Rank
{
    // Compute rank using SVD
    // Rank is the number of non-zero singular values (or > epsilon)
    Matrix<T> *U, *S, *Vt;
    this->svd(U, S, Vt);
    
    int r = 0;
    int k = std::min(rows_, columns_);
    for(int i=0; i<k; i++) {
        // Singular values are on diagonal of S
        // Use a relative tolerance based on max singular value
        // But for simplicity, simple epsilon check or max_sv * epsilon * max(m,n)
        double sigma = std::abs((double)S->get(i, i));
        if (sigma > 1e-10) { // A bit loose tolerance, or standard machine epsilon logic
             r++;
        }
    }
    
    delete U; delete S; delete Vt;
    return r;
}

/**********************************************
  * Algorithm part
  **********************************************/

// Calculate the pairwise interpoint Euclidean distances
// X is data matrix, D is distance matrix
void EuclideanDistanceMatrix(Matrix<double> *X, Matrix<double> *D)
{
  int i, j, k;
  double temp;
  if (D == NULL)
  {
    throw std::invalid_argument("Input matrix pointer is NULL.");
  }
  else if (X->rows() != D->rows() || X->rows() != D->columns())
  {
    throw std::invalid_argument("Invalid distance matrix dimension.");
  }

  for (i = 0; i < D->rows(); i++)
    D->set(i, i, 0.0);

  for (i = 0; i < D->rows() - 1; i++)
  {
    for (j = i + 1; j < D->columns(); j++)
    {
      temp = 0;
      for (k = 0; k < X->columns(); k++)
      {
        temp += pow(X->get(i, k) - X->get(j, k), 2);
      }
      D->set(i, j, sqrt(temp));
    }
  }

  for (i = 1; i < D->rows(); i++)
  {
    for (j = 0; j < i; j++)
    {
      D->set(i, j, D->get(j, i));
    }
  }
}

// Copy all elements of X to Y
void ElementCopy(Matrix<double> *X, Matrix<double> *Y)
{
  if (Y == NULL)
  {
    throw std::invalid_argument("Input matrix pointer is NULL.");
  }
  else if (X->rows() != Y->rows() || X->columns() != Y->columns())
  {
    throw std::invalid_argument("Invalid matrix dimension.");
  }
  for (int i = 0; i < X->rows(); i++)
  {
    for (int j = 0; j < X->columns(); j++)
    {
      Y->set(i, j, X->get(i, j));
    }
  }
}

// Multidimensional scaling (MDS)
// This function re-implements Laurens van der Maaten's MDS in his Matlab Toolbox for Dimensionality Reduction
// The Matlab MDS can be downloaded at http://crcv.ucf.edu/source/dimension
Matrix<double> *MDS_UCF(Matrix<double> *D, Matrix<double> *X0, int dim, int iter)
{
  if (D->rows() != D->columns())
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

  Matrix<double> *X = NULL;

  // with initialization
  if (X0 != NULL)
  {
    if (X0->rows() != D->rows() || X0->columns() != dim)
    {
      throw std::invalid_argument("Input initialization to MDS has invalid dimension.");
    }
    X = X0->copy();
  }
  // without initialization
  else
  {
    X = new Matrix<double>(D->rows(), dim, "rand");
    double D_mean = D->mean();                                             // mean value of distance matrix
    X->addNumberSelf(-0.5);                                                // move to the center
    X->multiplyNumberSelf(0.1 * D_mean / (1.0 / 3.0 * sqrt((double)dim))); // before this step, mean distance is 1/3*sqrt(d)
  }

  double lr = 0.05;  // learning rate
  double r = 2;      // metric
  int n = D->rows(); // number of vectors

  Matrix<double> *dh = new Matrix<double>(n, n, 0.0);
  Matrix<double> *pmat = new Matrix<double>(n, dim);
  Matrix<double> *dhdum = new Matrix<double>(n, 1);
  Matrix<double> *dhmat = new Matrix<double>(n - 1, dim, 0);

  Matrix<int> *RP = new Matrix<int>(n, iter, "randperm"); // the matrix for random permutation numbers
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
       // ... existing code ...
       m = RP->get(rp, it) - 1;

       for (i = 0; i < n; i++)
       {
         for (j = 0; j < dim; j++)
         {
           pmat->set(i, j, X->get(m, j) - X->get(i, j));
         }
       }

       for (i = 0; i < n; i++)
       {
         temp = 0;
         for (j = 0; j < dim; j++)
         {
           temp += pow(fabs(pmat->get(i, j)), r);
         }
         dhdum->set(i, 0, pow(temp, 1 / r));
       }

       for (i = 0; i < n; i++)
       {
         if (i == m)
           continue;

         dh->set(m, i, dhdum->get(i, 0));
         dh->set(i, m, dhdum->get(i, 0));
       }

       for (i = 0; i < n - 1; i++)
       {
         int ii = i;
         if (i >= m)
           ii = i + 1;
         temp = lr * (dhdum->get(ii, 0) - D->get(ii, m)) * pow(dhdum->get(ii, 0), 1 - r);
         for (j = 0; j < dim; j++)
         {
           dhmat->set(i, j, temp);
         }
       }

       for (i = 0; i < n - 1; i++)
       {
         int ii = i;
         if (i >= m)
           ii = i + 1;
         for (j = 0; j < dim; j++)
         {
           temp = X->get(ii, j);
           temp += dhmat->get(i, j) * pow(fabs(pmat->get(ii, j)), r - 1) * sign<double>(pmat->get(ii, j));

           X->set(ii, j, temp);
         }
       }
    }
  }

  printf("\n");

  delete dh;
  delete pmat;
  delete dhdum;
  delete dhmat;
  delete RP;

  return X;
}

// Multidimensional scaling (MDS) with SMACOF
// This code re-implements Michael Bronstein's SMACOF in his Matlab Toolbox for Surface Comparison and Analysis
// The Matlab SMACOF can be downloaded at http://tosca.cs.technion.ac.il/
Matrix<double> *MDS_SMACOF(Matrix<double> *D, Matrix<double> *X0, int dim, int iter)
{
  if (D->rows() != D->columns())
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

  Matrix<double> *X = NULL;

  // with initialization
  if (X0 != NULL)
  {
    if (X0->rows() != D->rows() || X0->columns() != dim)
    {
      throw std::invalid_argument("Input initialization to MDS has invalid dimension.");
    }
    X = X0->copy();
  }
  // without initialization
  else
  {
    X = new Matrix<double>(D->rows(), dim, "rand");
    double D_mean = D->mean();                                             // mean value of distance matrix
    X->addNumberSelf(-0.5);                                                // move to the center
    X->multiplyNumberSelf(0.1 * D_mean / (1.0 / 3.0 * sqrt((double)dim))); // before this step, mean distance is 1/3*sqrt(d)
  }

  Matrix<double> *Z = X->copy();
  Matrix<double> *D_ = new Matrix<double>(D->rows(), D->columns());
  Matrix<double> *B = new Matrix<double>(D->rows(), D->columns());
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
    for (i = 0; i < D->rows(); i++)
    {
      for (j = 0; j < D->columns(); j++)
      {
        if (i == j || fabs(D_->get(i, j)) < EPSILON)
        {
          B->set(i, j, 0.0);
        }
        else
        {
          B->set(i, j, -D->get(i, j) / D_->get(i, j));
        }
      }
    }

    for (j = 0; j < D->columns(); j++)
    {
      temp = 0;
      for (i = 0; i < D->rows(); i++)
      {
        temp += B->get(i, j);
      }
      B->set(j, j, -temp);
    }

    // X = B*Z/size(D,1);
    for (i = 0; i < X->rows(); i++)
    {
      for (j = 0; j < X->columns(); j++)
      {
        temp = 0;
        for (k = 0; k < B->columns(); k++)
        {
          temp += (B->get(i, k) * Z->get(k, j));
        }
        X->set(i, j, temp / (double)D->rows());
      }
    }

    // D_ = calc_D (X);
    EuclideanDistanceMatrix(X, D_);

    // Z = X;
    ElementCopy(X, Z);
  }

  printf("\n");

  delete Z;
  delete D_;
  delete B;

  return X;
}
} // namespace smat

#endif
