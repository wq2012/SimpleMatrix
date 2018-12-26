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

void test_int_matrix()
{
  smat::Matrix<double> *A = new smat::Matrix<double>(3, 4, 1);
  assertIntEq(A->get(2, 3), 1);

  // get, rows, columns, trace and sum
  A->set(2, 3, 5);
  assertIntEq(A->get(2, 3), 5);
  assertIntEq(A->get(1, 1), 1);
  assertIntEq(A->rows(), 3);
  assertIntEq(A->columns(), 4);
  assertIntEq(A->trace(), 3);
  assertIntEq(A->sum(), 16);

  // max and min
  int r, c;
  A->set(1, 1, -2);
  int max_el = A->maxEl(r, c);
  assertIntEq(max_el, 5);
  assertIntEq(r, 2);
  assertIntEq(c, 3);
  int min_el = A->minEl(r, c);
  assertIntEq(min_el, -2);
  assertIntEq(r, 1);
  assertIntEq(c, 1);

  // mean, fnorm, and pnorm
  assertFloatEqual(A->mean(), 1.08333);
  assertFloatEqual(A->fnorm(), 6.245);
  assertFloatEqual(A->pnorm(0.0), 12.0);
  assertFloatEqual(A->pnorm(1.0), 17.0);
  assertFloatEqual(A->pnorm(2.0), 6.245);

  delete A;
}

int main(int argc, const char *argv[])
{
  test_int_matrix();
  return 0;
}
