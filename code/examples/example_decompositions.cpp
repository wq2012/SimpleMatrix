#include <iostream>
#include "../SimpleMatrix.h"

int main() {
    std::cout << "============================================================\n";
    std::cout << "Example 2: Matrix Decompositions, Subspaces & Low-Rank SVD\n";
    std::cout << "============================================================\n\n";

    // 1. QR Decomposition on a rectangular matrix
    smat::Matrix<double> A = {
        {12.0, -51.0,   4.0},
        { 6.0, 167.0, -68.0},
        {-4.0,  24.0, -41.0},
        {-1.0,   1.0,   0.0}
    };
    smat::Matrix<double> Q, R;
    A.qr(Q, R);
    std::cout << "4x3 Matrix A:\n" << A;
    std::cout << "Orthogonal Q (4x4):\n" << Q;
    std::cout << "Upper Trapezoidal R (4x3):\n" << R;
    std::cout << "Reconstruction error ||Q*R - A||_F = " << (Q * R - A).fnorm() << "\n\n";

    // 2. SVD & Rank-1 Approximation
    smat::Matrix<double> U, S, Vt;
    A.svd(U, S, Vt);
    std::cout << "Singular Values of A (diagonal of S):\n" << S.diag().transpose();

    // Rank-1 approximation: sigma_0 * u_0 * v_0^T
    smat::Matrix<double> A_rank1 = S(0, 0) * (U.col(0) * Vt.row(0));
    std::cout << "\nRank-1 Approximation of A:\n" << A_rank1;
    std::cout << "Rank of A_rank1 = " << A_rank1.rank() << "\n\n";

    // 3. Column Space (orth) and Null Space (nullSpace) of a Rank-Deficient Matrix
    smat::Matrix<double> RankDef = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    smat::Matrix<double> ColBasis = RankDef.orth();
    smat::Matrix<double> NullBasis = RankDef.nullSpace();
    std::cout << "Rank-Deficient 3x3 Matrix (rank = " << RankDef.rank() << "):\n" << RankDef;
    std::cout << "Orthonormal Column Space Basis (3x2):\n" << ColBasis;
    std::cout << "Orthonormal Null Space Basis (3x1):\n" << NullBasis;
    std::cout << "Verification ||RankDef * NullBasis||_F = " << (RankDef * NullBasis).fnorm() << "\n";

    return 0;
}
