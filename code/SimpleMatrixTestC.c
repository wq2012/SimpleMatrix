#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "SimpleMatrix.h"

#define TEST_EPS 1e-6

static void assert_int_eq(int a, int b, const char *msg) {
    if (a != b) {
        fprintf(stderr, "[FAIL] %s: expected %d, got %d\n", msg, b, a);
        exit(1);
    }
}

static void assert_dbl_eq(double a, double b, double tol, const char *msg) {
    if (fabs(a - b) > tol) {
        fprintf(stderr, "[FAIL] %s: expected %.8f, got %.8f (tol=%.8f)\n", msg, b, a, tol);
        exit(1);
    }
}

static void assert_true(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "[FAIL] %s\n", msg);
        exit(1);
    }
}

static double square_fn(double x) {
    return x * x;
}

static void test_c_creation_and_access(void) {
    printf("[C TEST] Creation & Access...\n");
    smat_t *Z = smat_zeros(3, 4);
    assert_true(Z != NULL, "smat_zeros non-null");
    assert_int_eq(smat_rows(Z), 3, "rows");
    assert_int_eq(smat_cols(Z), 4, "cols");
    assert_int_eq(smat_size(Z), 12, "size");
    assert_dbl_eq(smat_sum(Z), 0.0, TEST_EPS, "zeros sum");

    assert_int_eq(smat_set(Z, 1, 2, 7.5), SMAT_OK, "smat_set valid");
    assert_dbl_eq(smat_get(Z, 1, 2), 7.5, TEST_EPS, "smat_get valid");
    assert_dbl_eq(Z->v[1][2], 7.5, TEST_EPS, "direct v[r][c] access");
    assert_int_eq(smat_set(Z, 10, 0, 1.0), SMAT_ERR_BOUNDS, "smat_set out of bounds");

    smat_t *O = smat_ones(2, 3);
    assert_dbl_eq(smat_sum(O), 6.0, TEST_EPS, "ones sum");

    smat_t *I = smat_identity(4);
    assert_true(smat_is_square(I), "identity square");
    assert_true(smat_is_symmetric(I, 1e-9), "identity symmetric");
    assert_dbl_eq(smat_trace(I), 4.0, TEST_EPS, "identity trace");

    /* Random matrices in same second must differ */
    smat_t *R1 = smat_rand(4, 4);
    smat_t *R2 = smat_rand(4, 4);
    assert_true(!smat_equals(R1, R2, 1e-9), "consecutive smat_rand must differ");

    /* Seeded random matrices must be reproducible */
    smat_seed(12345);
    smat_t *S1 = smat_randn(3, 3);
    smat_seed(12345);
    smat_t *S2 = smat_randn(3, 3);
    assert_true(smat_equals(S1, S2, 1e-12), "seeded smat_randn reproducible");

    /* Linspace & arange & diag */
    smat_t *L = smat_linspace(0.0, 1.0, 5);
    assert_int_eq(L->cols, 5, "linspace cols");
    assert_dbl_eq(L->data[2], 0.5, TEST_EPS, "linspace midpoint");

    smat_t *Ar = smat_arange(1.0, 5.0, 1.0);
    assert_int_eq(Ar->cols, 4, "arange cols");
    assert_dbl_eq(smat_sum(Ar), 10.0, TEST_EPS, "arange sum 1+2+3+4");

    smat_t *D = smat_diag(Ar);
    assert_int_eq(D->rows, 4, "diag matrix rows");
    assert_int_eq(D->cols, 4, "diag matrix cols");
    assert_dbl_eq(smat_trace(D), 10.0, TEST_EPS, "diag matrix trace");
    smat_t *d_vec = smat_diag(D);
    assert_int_eq(d_vec->rows, 4, "extracted diag rows");
    assert_int_eq(d_vec->cols, 1, "extracted diag cols");
    assert_dbl_eq(smat_sum(d_vec), 10.0, TEST_EPS, "extracted diag sum");

    smat_free(Z); smat_free(O); smat_free(I);
    smat_free(R1); smat_free(R2); smat_free(S1); smat_free(S2);
    smat_free(L); smat_free(Ar); smat_free(D); smat_free(d_vec);
    smat_free(NULL); /* Safe on NULL */
}

static void test_c_structural_and_io(void) {
    printf("[C TEST] Structural, Slicing & I/O...\n");
    double vals[6] = {1, 2, 3, 4, 5, 6};
    smat_t *A = smat_from_array(2, 3, vals);
    smat_t *At = smat_transpose(A);
    assert_int_eq(At->rows, 3, "transpose rows");
    assert_int_eq(At->cols, 2, "transpose cols");
    assert_dbl_eq(At->v[2][1], 6.0, TEST_EPS, "transpose element");

    smat_t *R = smat_reshape(A, 3, 2);
    assert_int_eq(R->rows, 3, "reshape rows");
    assert_dbl_eq(R->v[2][1], 6.0, TEST_EPS, "reshape element");

    smat_t *Sub = smat_sub(A, 0, 1, 1, 2);
    assert_int_eq(Sub->rows, 2, "sub rows");
    assert_int_eq(Sub->cols, 2, "sub cols");
    assert_dbl_eq(Sub->v[0][0], 2.0, TEST_EPS, "sub(0,0)");
    assert_dbl_eq(Sub->v[1][1], 6.0, TEST_EPS, "sub(1,1)");

    smat_t *Row0 = smat_row(A, 0);
    smat_t *Col2 = smat_col(A, 2);
    assert_dbl_eq(smat_sum(Row0), 6.0, TEST_EPS, "row 0 sum (1+2+3)");
    assert_dbl_eq(smat_sum(Col2), 9.0, TEST_EPS, "col 2 sum (3+6)");

    smat_t *H = smat_concat_right(A, A);
    smat_t *V = smat_concat_bottom(A, A);
    assert_int_eq(H->cols, 6, "concat_right cols");
    assert_int_eq(V->rows, 4, "concat_bottom rows");

    smat_t *U = smat_triu(At, 0);
    smat_t *Lo = smat_tril(At, 0);
    assert_dbl_eq(U->v[1][0], 0.0, TEST_EPS, "triu below diag is 0");
    assert_dbl_eq(Lo->v[0][1], 0.0, TEST_EPS, "tril above diag is 0");

    /* Save and load txt */
    const char *tmp_file = "c_test_matrix_io.txt";
    assert_int_eq(smat_save_txt(A, tmp_file), SMAT_OK, "save_txt");
    smat_t *Loaded = smat_load_txt(tmp_file);
    assert_true(Loaded != NULL, "load_txt non-null");
    assert_true(smat_equals(A, Loaded, 1e-5), "loaded equals saved");
    remove(tmp_file);

    smat_free(A); smat_free(At); smat_free(R); smat_free(Sub);
    smat_free(Row0); smat_free(Col2); smat_free(H); smat_free(V);
    smat_free(U); smat_free(Lo); smat_free(Loaded);
}

static void test_c_arithmetic_and_products(void) {
    printf("[C TEST] Arithmetic & Vector/Matrix Products...\n");
    double a_vals[4] = {1, 2, 3, 4};
    double b_vals[4] = {5, 6, 7, 8};
    smat_t *A = smat_from_array(2, 2, a_vals);
    smat_t *B = smat_from_array(2, 2, b_vals);

    smat_t *Sum = smat_add(A, B);
    smat_t *Diff = smat_sub_mat(B, A);
    smat_t *DotM = smat_dot_mul(A, B);
    smat_t *DotD = smat_dot_div(B, A);
    smat_t *Prod = smat_mul(A, B);

    assert_dbl_eq(Sum->v[0][0], 6.0, TEST_EPS, "A+B");
    assert_dbl_eq(Diff->v[1][1], 4.0, TEST_EPS, "B-A");
    assert_dbl_eq(DotM->v[1][0], 21.0, TEST_EPS, "A.*B");
    assert_dbl_eq(DotD->v[1][1], 2.0, TEST_EPS, "B./A");
    /* [1 2; 3 4] * [5 6; 7 8] = [19 22; 43 50] */
    assert_dbl_eq(Prod->v[0][0], 19.0, TEST_EPS, "(A*B)(0,0)");
    assert_dbl_eq(Prod->v[0][1], 22.0, TEST_EPS, "(A*B)(0,1)");
    assert_dbl_eq(Prod->v[1][0], 43.0, TEST_EPS, "(A*B)(1,0)");
    assert_dbl_eq(Prod->v[1][1], 50.0, TEST_EPS, "(A*B)(1,1)");

    smat_t *Sq = smat_apply(A, square_fn);
    assert_dbl_eq(Sq->v[1][1], 16.0, TEST_EPS, "smat_apply square");

    /* 3D Cross, Dot, Outer, Kronecker */
    double u_vals[3] = {1, 0, 0};
    double v_vals[3] = {0, 1, 0};
    smat_t *u = smat_from_array(3, 1, u_vals);
    smat_t *v = smat_from_array(3, 1, v_vals);
    assert_dbl_eq(smat_dot(u, v), 0.0, TEST_EPS, "orthogonal dot product");
    smat_t *w = smat_cross(u, v);
    assert_dbl_eq(w->data[0], 0.0, TEST_EPS, "cross x");
    assert_dbl_eq(w->data[1], 0.0, TEST_EPS, "cross y");
    assert_dbl_eq(w->data[2], 1.0, TEST_EPS, "cross z");

    smat_t *Out = smat_outer(u, v);
    assert_int_eq(Out->rows, 3, "outer rows");
    assert_int_eq(Out->cols, 3, "outer cols");
    assert_dbl_eq(Out->v[0][1], 1.0, TEST_EPS, "outer(0,1)");

    smat_t *I2 = smat_identity(2);
    smat_t *K = smat_kron(I2, A);
    assert_int_eq(K->rows, 4, "kron rows");
    assert_int_eq(K->cols, 4, "kron cols");
    assert_dbl_eq(smat_trace(K), 2.0 * smat_trace(A), TEST_EPS, "trace(I2 kron A)");

    smat_free(A); smat_free(B); smat_free(Sum); smat_free(Diff);
    smat_free(DotM); smat_free(DotD); smat_free(Prod); smat_free(Sq);
    smat_free(u); smat_free(v); smat_free(w); smat_free(Out);
    smat_free(I2); smat_free(K);
}

static void test_c_stats_and_norms(void) {
    printf("[C TEST] Statistics, Reductions & Norms...\n");
    double vals[6] = {1, 2, 3, 4, 5, 6};
    smat_t *A = smat_from_array(2, 3, vals);

    int mr = -1, mc = -1;
    assert_dbl_eq(smat_max(A, &mr, &mc), 6.0, TEST_EPS, "max val");
    assert_int_eq(mr, 1, "max row");
    assert_int_eq(mc, 2, "max col");
    assert_dbl_eq(smat_min(A, &mr, &mc), 1.0, TEST_EPS, "min val");
    assert_int_eq(mr, 0, "min row");
    assert_int_eq(mc, 0, "min col");

    smat_t *S0 = smat_sum_axis(A, 0);
    smat_t *S1 = smat_sum_axis(A, 1);
    assert_dbl_eq(S0->v[0][0], 5.0, TEST_EPS, "col 0 sum (1+4)");
    assert_dbl_eq(S1->v[0][0], 6.0, TEST_EPS, "row 0 sum (1+2+3)");
    assert_dbl_eq(S1->v[1][0], 15.0, TEST_EPS, "row 1 sum (4+5+6)");

    assert_dbl_eq(smat_norm_1(A), 9.0, TEST_EPS, "1-norm max col sum (3+6)");
    assert_dbl_eq(smat_norm_inf(A), 15.0, TEST_EPS, "inf-norm max row sum (4+5+6)");

    /* Covariance and correlation */
    double x_vals[6] = {1, 2, 2, 4, 3, 6}; /* col 1 is 2 * col 0 -> corr = 1 */
    smat_t *X = smat_from_array(3, 2, x_vals);
    smat_t *Cov = smat_cov(X);
    smat_t *Corr = smat_corrcoef(X);
    assert_dbl_eq(Cov->v[0][0], 1.0, TEST_EPS, "var(col 0) = 1");
    assert_dbl_eq(Cov->v[1][1], 4.0, TEST_EPS, "var(col 1) = 4");
    assert_dbl_eq(Corr->v[0][1], 1.0, TEST_EPS, "perfect positive correlation");

    smat_free(A); smat_free(S0); smat_free(S1);
    smat_free(X); smat_free(Cov); smat_free(Corr);
}

static void test_c_decompositions_and_solvers(void) {
    printf("[C TEST] Decompositions (LU, Cholesky, QR, Eigen, SVD) & Solvers...\n");

    /* 1. LU on non-singular and singular matrices */
    double a_vals[9] = {1, 2, 3, 4, 5, 6, 7, 8, 7};
    smat_t *A = smat_from_array(3, 3, a_vals);
    assert_dbl_eq(smat_determinant(A), 6.0, TEST_EPS, "det(A) = 6");

    smat_t *L = NULL, *U = NULL, *P = NULL;
    assert_int_eq(smat_lu(A, &L, &U, &P), SMAT_OK, "smat_lu");
    smat_t *PA = smat_mul(P, A);
    smat_t *LU = smat_mul(L, U);
    assert_true(smat_equals(PA, LU, TEST_EPS), "PA == LU");
    smat_free(L); smat_free(U); smat_free(P); smat_free(PA); smat_free(LU);

    /* Singular LU must not produce NaN */
    double sing_vals[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    smat_t *Sing = smat_from_array(3, 3, sing_vals);
    assert_int_eq(smat_lu(Sing, &L, &U, &P), SMAT_OK, "singular smat_lu");
    PA = smat_mul(P, Sing);
    LU = smat_mul(L, U);
    assert_true(smat_equals(PA, LU, TEST_EPS), "singular PA == LU");
    smat_free(L); smat_free(U); smat_free(P); smat_free(PA); smat_free(LU);

    /* 2. Cholesky on SPD matrix */
    double spd_vals[9] = {4, 12, -16, 12, 37, -43, -16, -43, 98};
    smat_t *SPD = smat_from_array(3, 3, spd_vals);
    smat_t *CholL = smat_cholesky(SPD);
    assert_true(CholL != NULL, "Cholesky L non-null");
    smat_t *CholLt = smat_transpose(CholL);
    smat_t *LLt = smat_mul(CholL, CholLt);
    assert_true(smat_equals(SPD, LLt, TEST_EPS), "Cholesky L * L^T == SPD");
    smat_free(CholL); smat_free(CholLt); smat_free(LLt);

    /* 3. QR on square, tall, and wide matrices */
    smat_t *Q = NULL, *R = NULL;
    assert_int_eq(smat_qr(A, &Q, &R), SMAT_OK, "smat_qr square");
    smat_t *QR = smat_mul(Q, R);
    assert_true(smat_equals(A, QR, TEST_EPS), "Q * R == A");
    smat_free(Q); smat_free(R); smat_free(QR);

    double wide_vals[6] = {1, 2, 3, 4, 5, 6};
    smat_t *Wide = smat_from_array(2, 3, wide_vals);
    assert_int_eq(smat_qr(Wide, &Q, &R), SMAT_OK, "smat_qr wide (m < n)");
    QR = smat_mul(Q, R);
    assert_true(smat_equals(Wide, QR, TEST_EPS), "wide Q * R == Wide");
    smat_free(Q); smat_free(R); smat_free(QR);

    /* 4. Symmetric Eigen decomposition (sorted descending) */
    smat_t *V = NULL, *D = NULL;
    assert_int_eq(smat_eigen(SPD, &V, &D), SMAT_OK, "smat_eigen");
    assert_true(D->v[0][0] >= D->v[1][1] && D->v[1][1] >= D->v[2][2], "eigenvalues sorted descending");
    smat_t *AV = smat_mul(SPD, V);
    smat_t *VD = smat_mul(V, D);
    assert_true(smat_equals(AV, VD, TEST_EPS), "A * V == V * D");
    smat_free(V); smat_free(D); smat_free(AV); smat_free(VD);

    /* 5. SVD on rank-deficient matrix (verify orthonormal U and V) */
    smat_t *U_svd = NULL, *S_svd = NULL, *Vt_svd = NULL;
    assert_int_eq(smat_svd(Sing, &U_svd, &S_svd, &Vt_svd), SMAT_OK, "smat_svd rank-deficient");
    assert_true(S_svd->v[0][0] >= S_svd->v[1][1] && S_svd->v[1][1] >= S_svd->v[2][2], "singular values sorted descending");
    smat_t *US = smat_mul(U_svd, S_svd);
    smat_t *USVt = smat_mul(US, Vt_svd);
    assert_true(smat_equals(Sing, USVt, TEST_EPS), "SVD U * S * Vt == Sing");
    smat_t *Ut = smat_transpose(U_svd);
    smat_t *UtU = smat_mul(Ut, U_svd);
    smat_t *I3 = smat_identity(3);
    assert_true(smat_equals(UtU, I3, TEST_EPS), "Rank-deficient SVD U^T * U == I");
    smat_free(U_svd); smat_free(S_svd); smat_free(Vt_svd);
    smat_free(US); smat_free(USVt); smat_free(Ut); smat_free(UtU);

    /* 6. Pseudoinverse Moore-Penrose 4 conditions */
    smat_t *Pinv = smat_pinv(Sing);
    smat_t *AP = smat_mul(Sing, Pinv);
    smat_t *PA_pinv = smat_mul(Pinv, Sing);
    smat_t *APA = smat_mul(AP, Sing);
    smat_t *PAP = smat_mul(PA_pinv, Pinv);
    assert_true(smat_equals(APA, Sing, TEST_EPS), "Moore-Penrose 1: A * A+ * A == A");
    assert_true(smat_equals(PAP, Pinv, TEST_EPS), "Moore-Penrose 2: A+ * A * A+ == A+");
    assert_true(smat_is_symmetric(AP, TEST_EPS), "Moore-Penrose 3: (A * A+)^T == A * A+");
    assert_true(smat_is_symmetric(PA_pinv, TEST_EPS), "Moore-Penrose 4: (A+ * A)^T == A+ * A");
    smat_free(Pinv); smat_free(AP); smat_free(PA_pinv); smat_free(APA); smat_free(PAP);

    /* 7. Solve, Least Squares, Null Space, Orth, Expm */
    double b_vals[3] = {6, 15, 22};
    smat_t *b = smat_from_array(3, 1, b_vals);
    smat_t *x = smat_solve(A, b);
    smat_t *Ax = smat_mul(A, x);
    assert_true(smat_equals(Ax, b, TEST_EPS), "A * x == b");
    smat_free(b); smat_free(x); smat_free(Ax);

    /* Null space of rank-2 3x3 matrix Sing has 1 column and Sing * N == 0 */
    assert_int_eq(smat_rank(Sing), 2, "rank(Sing) == 2");
    smat_t *N = smat_null_space(Sing);
    assert_true(N != NULL && N->cols == 1, "null_space cols == 1");
    smat_t *SingN = smat_mul(Sing, N);
    assert_dbl_eq(smat_fnorm(SingN), 0.0, TEST_EPS, "Sing * null_space == 0");
    smat_t *Q_orth = smat_orth(Sing);
    assert_true(Q_orth != NULL && Q_orth->cols == 2, "orth cols == 2");
    smat_free(N); smat_free(SingN); smat_free(Q_orth);

    /* Matrix exponential: expm(0) == I, and expm(A) * expm(-A) == I */
    smat_t *NegA = smat_neg(A);
    smat_t *ExpA = smat_expm(A);
    smat_t *ExpNegA = smat_expm(NegA);
    smat_t *ExpProd = smat_mul(ExpA, ExpNegA);
    assert_true(smat_equals(ExpProd, I3, 1e-5), "expm(A) * expm(-A) == I");
    smat_free(NegA); smat_free(ExpA); smat_free(ExpNegA); smat_free(ExpProd);

    /* 8. PCA & Classical MDS */
    smat_t *Proj = NULL, *Comp = NULL, *ExpVar = NULL;
    assert_int_eq(smat_pca(SPD, 2, &Proj, &Comp, &ExpVar), SMAT_OK, "smat_pca");
    assert_int_eq(Proj->cols, 2, "pca projected cols");
    assert_int_eq(Comp->cols, 2, "pca components cols");
    smat_free(Proj); smat_free(Comp); smat_free(ExpVar);

    /* Classical MDS on unit square */
    double sq_pts[8] = {0, 0, 0, 1, 1, 0, 1, 1};
    smat_t *Pts = smat_from_array(4, 2, sq_pts);
    smat_t *Dist = smat_pdist(Pts);
    smat_t *RecPts = smat_mds_classical(Dist, 2);
    smat_t *RecDist = smat_pdist(RecPts);
    assert_true(smat_equals(Dist, RecDist, TEST_EPS), "Classical MDS preserves Euclidean distances exactly");

    smat_free(Pts); smat_free(Dist); smat_free(RecPts); smat_free(RecDist);
    smat_free(A); smat_free(Sing); smat_free(SPD); smat_free(Wide); smat_free(I3);
}

int main(void) {
    printf("============================================================\n");
    printf("Running SimpleMatrix Pure C Test Suite\n");
    printf("============================================================\n");
    test_c_creation_and_access();
    test_c_structural_and_io();
    test_c_arithmetic_and_products();
    test_c_stats_and_norms();
    test_c_decompositions_and_solvers();
    printf("============================================================\n");
    printf("All Pure C tests passed!\n");
    printf("============================================================\n");
    return 0;
}
