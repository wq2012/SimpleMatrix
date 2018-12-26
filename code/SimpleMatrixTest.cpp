#include <cassert>
#include "SimpleMatrix.h"

using namespace std;

void test_get_set()
{
  smat::Matrix<double> *A = new smat::Matrix<double>(3, 4, 1);
  assert(A->get(2, 3) == 1);
  A->set(2, 3, 5);
  assert(A->get(2, 3) == 5);
  assert(A->get(1, 1) == 1);
  delete A;
}

int main(int argc, const char *argv[])
{
  test_get_set();
  return 0;
}
