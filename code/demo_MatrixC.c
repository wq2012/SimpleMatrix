#include <stdio.h>
#include "SimpleMatrix.h"

int main(void) {
    printf("============================================================\n");
    printf("SimpleMatrix Pure C Demo\n");
    printf("============================================================\n");

    /* 1. Matrix creation and basic operations */
    double a_data[9] = {
        4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
        -16.0, -43.0, 98.0
    };
    smat_t *A = smat_from_array(3, 3, a_data);
    printf("\nSymmetric Positive-Definite Matrix A:\n");
    smat_print(A);

    printf("Trace(A) = %.4f, Det(A) = %.4f, Rank(A) = %d, Cond(A) = %.4f\n",
           smat_trace(A), smat_determinant(A), smat_rank(A), smat_cond(A));

    /* 2. Cholesky Decomposition: A = L * L^T */
    smat_t *L = smat_cholesky(A);
    printf("\nCholesky Lower Triangular Factor L (A = L * L^T):\n");
    smat_print(L);

    /* 3. Solving a linear system A * x = b */
    double b_data[3] = {1.0, 2.0, 3.0};
    smat_t *b = smat_from_array(3, 1, b_data);
    smat_t *x = smat_solve(A, b);
    printf("\nSolution x to A * x = [1; 2; 3]:\n");
    smat_print(x);

    /* 4. Symmetric Eigenvalue Decomposition: A * V = V * D */
    smat_t *V = NULL, *D = NULL;
    smat_eigen(A, &V, &D);
    printf("\nEigenvalues of A (sorted descending):\n");
    smat_t *eig = smat_diag(D);
    smat_print(eig);

    /* 5. Singular Value Decomposition (SVD) */
    smat_t *U = NULL, *S = NULL, *Vt = NULL;
    smat_svd(A, &U, &S, &Vt);
    printf("\nSingular Values of A:\n");
    smat_t *sv = smat_diag(S);
    smat_print(sv);

    /* 6. Classical Multidimensional Scaling (MDS) */
    double pts_data[8] = {
        0.0, 0.0,
        0.0, 2.0,
        3.0, 0.0,
        3.0, 2.0
    };
    smat_t *Pts = smat_from_array(4, 2, pts_data);
    smat_t *Dist = smat_pdist(Pts);
    smat_t *Embedded = smat_mds_classical(Dist, 2);
    printf("\nPairwise Distance Matrix of 4 Rectangle Corners:\n");
    smat_print(Dist);
    printf("\nClassical MDS 2D Embedding:\n");
    smat_print(Embedded);

    /* Clean up all allocated matrices */
    smat_free(A);
    smat_free(L);
    smat_free(b);
    smat_free(x);
    smat_free(V);
    smat_free(D);
    smat_free(eig);
    smat_free(U);
    smat_free(S);
    smat_free(Vt);
    smat_free(sv);
    smat_free(Pts);
    smat_free(Dist);
    smat_free(Embedded);

    printf("\n[DONE] Pure C Demo completed successfully.\n");
    return 0;
}
