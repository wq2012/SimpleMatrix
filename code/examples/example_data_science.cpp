#include <iostream>
#include "../SimpleMatrix.h"

int main() {
    std::cout << "============================================================\n";
    std::cout << "Example 3: Data Science (Least Squares, Covariance, PCA, MDS)\n";
    std::cout << "============================================================\n\n";

    // 1. Polynomial Curve Fitting via Least Squares (lstsq)
    // Fit quadratic y = c0 + c1*x + c2*x^2 to noisy observations of y = 1 + 2x + 0.5x^2
    smat::Matrix<double> x = smat::Matrix<double>::linspace(0.0, 4.0, 5).transpose();
    smat::Matrix<double> X_design(5, 3);
    smat::Matrix<double> y(5, 1);
    for (int i = 0; i < 5; ++i) {
        double xi = x(i, 0);
        X_design(i, 0) = 1.0;
        X_design(i, 1) = xi;
        X_design(i, 2) = xi * xi;
        y(i, 0) = 1.0 + 2.0 * xi + 0.5 * xi * xi;
    }
    smat::Matrix<double> coeffs = X_design.lstsq(y);
    std::cout << "Quadratic Regression Coefficients [c0, c1, c2]^T (Expected [1.0, 2.0, 0.5]):\n";
    std::cout << coeffs.transpose() << "\n";

    // 2. Multivariate Data Analysis: Covariance, Correlation & PCA
    smat::Matrix<double> Data = {
        {2.5, 2.4, 0.5},
        {0.5, 0.7, 0.1},
        {2.2, 2.9, 0.6},
        {1.9, 2.2, 0.4},
        {3.1, 3.0, 0.7},
        {2.3, 2.7, 0.5},
        {2.0, 1.6, 0.3},
        {1.0, 1.1, 0.2},
        {1.5, 1.6, 0.3},
        {1.1, 0.9, 0.2}
    };
    std::cout << "Sample Covariance Matrix (3x3):\n" << Data.cov();
    std::cout << "Pearson Correlation Matrix (3x3):\n" << Data.corrcoef();

    smat::Matrix<double> Projected, Components, ExplainedVar;
    Data.pca(2, Projected, Components, ExplainedVar);
    std::cout << "Top-2 PCA Principal Components (Loadings):\n" << Components;
    std::cout << "Explained Variance of Top-2 Components:\n" << ExplainedVar.transpose();
    std::cout << "2D PCA Projected Data (10x2):\n" << Projected << "\n";

    // 3. Classical (Torgerson) MDS from Pairwise Distances
    smat::Matrix<double> D(Data.rows(), Data.rows());
    smat::EuclideanDistanceMatrix(Data, D);
    smat::Matrix<double> MDS_2D = smat::MDS_Classical(D, 2);
    std::cout << "Classical MDS 2D Embedding (10x2):\n" << MDS_2D;

    return 0;
}
