#include <stdio.h>
#include "../SimpleMatrixC.h"

int main(void) {
    printf("============================================================\n");
    printf("Example 4: Pure C Scientific Computing with SimpleMatrixC.h\n");
    printf("============================================================\n\n");

    /* 1. Linear Least-Squares Regression in Pure C: y = m * x + b */
    double design_arr[8] = {
        1.0, 1.0,
        2.0, 1.0,
        3.0, 1.0,
        4.0, 1.0
    };
    double target_arr[4] = {3.1, 4.9, 7.0, 9.0};
    smat_t *X = smat_from_array(4, 2, design_arr);
    smat_t *y = smat_from_array(4, 1, target_arr);
    smat_t *theta = smat_lstsq(X, y);

    printf("1. Linear Regression Fit (y = slope * x + intercept):\n");
    printf("   Estimated [slope, intercept]: [%.4f, %.4f]\n\n",
           smat_get(theta, 0, 0), smat_get(theta, 1, 0));

    /* 2. Continuous-Time Markov Rate Matrix Exponential P(t) = expm(Q * t) */
    double q_arr[4] = {
        -0.5,  0.5,
         0.5, -0.5
    };
    smat_t *Q = smat_from_array(2, 2, q_arr);
    smat_t *P_t = smat_expm(Q);
    printf("2. Transition Matrix P(t=1) = expm(Q):\n");
    smat_print(P_t);

    /* 3. Moore-Penrose Pseudoinverse & Null Space in Pure C */
    double rank_def_arr[9] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        5.0, 7.0, 9.0
    };
    smat_t *M = smat_from_array(3, 3, rank_def_arr);
    smat_t *M_pinv = smat_pinv(M);
    smat_t *M_null = smat_null_space(M);
    printf("\n3. Rank-Deficient Matrix M (rank = %d):\n", smat_rank(M));
    printf("   Moore-Penrose Pseudoinverse M+:\n");
    smat_print(M_pinv);
    printf("   Orthonormal Null Space Basis of M:\n");
    smat_print(M_null);

    smat_free(X);
    smat_free(y);
    smat_free(theta);
    smat_free(Q);
    smat_free(P_t);
    smat_free(M);
    smat_free(M_pinv);
    smat_free(M_null);
    return 0;
}
