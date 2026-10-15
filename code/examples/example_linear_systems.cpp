#include <iostream>
#include "../SimpleMatrix.h"

int main() {
    std::cout << "============================================================\n";
    std::cout << "Example 1: Linear Systems, Cholesky, LU & Matrix Exponential\n";
    std::cout << "============================================================\n\n";

    // 1. Solve a 3x3 linear system A * x = b
    smat::Matrix<double> A = {
        { 3.0,  2.0, -1.0},
        { 2.0, -2.0,  4.0},
        {-1.0,  0.5, -1.0}
    };
    smat::Matrix<double> b = {{1.0}, {-2.0}, {0.0}};

    smat::Matrix<double> x = A.solve(b);
    std::cout << "Coefficient Matrix A:\n" << A;
    std::cout << "Right-hand side b:\n" << b;
    std::cout << "Solution x = A.solve(b):\n" << x;
    std::cout << "Residual ||A*x - b||_F = " << (A * x - b).fnorm() << "\n\n";

    // 2. LU Decomposition with Partial Pivoting: P * A = L * U
    smat::Matrix<double> L, U, P;
    A.lu(L, U, P);
    std::cout << "LU Decomposition (P * A = L * U):\n";
    std::cout << "Permutation P:\n" << P;
    std::cout << "Lower Triangular L:\n" << L;
    std::cout << "Upper Triangular U:\n" << U << "\n";

    // 3. Cholesky Decomposition on a Symmetric Positive-Definite (SPD) matrix
    smat::Matrix<double> SPD = {
        { 4.0, 12.0, -16.0},
        {12.0, 37.0, -43.0},
        {-16.0, -43.0, 98.0}
    };
    smat::Matrix<double> CholL = SPD.cholesky();
    std::cout << "SPD Matrix:\n" << SPD;
    std::cout << "Cholesky Factor L (SPD = L * L^T):\n" << CholL;
    std::cout << "Verification ||L * L^T - SPD||_F = " << (CholL * CholL.transpose() - SPD).fnorm() << "\n\n";

    // 4. Matrix Exponential expm(M) for solving dx/dt = M * x
    smat::Matrix<double> M = {
        { 0.0, -1.0},
        { 1.0,  0.0}
    }; // 90-degree rotation generator (pi/2 rad)
    double pi_over_2 = 3.14159265358979323846 / 2.0;
    smat::Matrix<double> Rot = (pi_over_2 * M).expm();
    std::cout << "Rotation Matrix exp(pi/2 * [0 -1; 1 0]):\n" << Rot;

    return 0;
}
