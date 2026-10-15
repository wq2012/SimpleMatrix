/***************************************************************
 *  Copyright (C) 2013-2026 Quan Wang <wangq10@rpi.edu>
 *  Signal Analysis and Machine Perception Laboratory
 *  Department of Electrical, Computer, and Systems Engineering
 *  Rensselaer Polytechnic Institute, Troy, NY 12180, USA
 *
 *  Related publication:
 *  Quan Wang, Kim L. Boyer.
 *  Feature Learning by Multidimensional Scaling and its Applications in Object Recognition.
 *  2013 26th SIBGRAPI Conference on Graphics, Patterns and Images (Sibgrapi). IEEE, 2013.
 *
 *  SimpleMatrix.h:
 *  Single-header matrix & linear algebra library for BOTH pure C (C99/C11)
 *  and C++ (C++11/14/17/20/23).
 ***************************************************************/

#ifndef SIMPLE_MATRIX_H
#define SIMPLE_MATRIX_H

#ifndef MAX_LINE_LENGTH
#define MAX_LINE_LENGTH 100000
#endif

#ifndef EPSILON
#define EPSILON 0.0000001
#endif

#define SMAT_INTERNAL_EPS 1e-12
#define SMAT_JACOBI_EPS   1e-14

/* ============================================================================
 * PART 1: PURE C API (Compatible with both C99/C11 and C++)
 * ============================================================================ */

#ifdef __cplusplus
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
extern "C" {
#else
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#endif

/**
 * Error codes for the Pure C API.
 */
typedef enum {
    SMAT_OK = 0,
    SMAT_ERR_NULL = -1,
    SMAT_ERR_ALLOC = -2,
    SMAT_ERR_DIM = -3,
    SMAT_ERR_BOUNDS = -4,
    SMAT_ERR_SINGULAR = -5,
    SMAT_ERR_NOT_SYMMETRIC = -6,
    SMAT_ERR_NOT_POS_DEF = -7,
    SMAT_ERR_IO = -8,
    SMAT_ERR_INVALID_ARG = -9
} smat_err_t;

/**
 * Pure C Matrix structure.
 * Stores elements in contiguous row-major order in `data`,
 * with row pointers `v[i] = &data[i * cols]` for convenient `A->v[i][j]` access.
 */
typedef struct {
    int rows;
    int cols;
    double *data;
    double **v;
} smat_t;

typedef smat_t SMatrix;

/* Internal PRNG state for C API so we never reset global srand() repeatedly */
static inline unsigned long long* smat_c_rng_state_(void) {
    static unsigned long long state = 88172645463325252ULL;
    static int initialized = 0;
    if (!initialized) {
        unsigned long long t = (unsigned long long)time(NULL);
        if (t != 0) state ^= (t * 6364136223846793005ULL + 1442695040888963407ULL);
        if (state == 0) state = 88172645463325252ULL;
        initialized = 1;
    }
    return &state;
}

static inline void smat_seed(unsigned int seed) {
    unsigned long long *st = smat_c_rng_state_();
    *st = (seed == 0) ? 88172645463325252ULL : (unsigned long long)seed * 6364136223846793005ULL + 1ULL;
}

static inline unsigned long long smat_c_rand_u64_(void) {
    unsigned long long *st = smat_c_rng_state_();
    unsigned long long x = *st;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *st = x;
    return x * 2685821657736338717ULL;
}

static inline double smat_c_rand_uniform_(void) {
    return (double)(smat_c_rand_u64_() >> 11) * (1.0 / 9007199254740992.0);
}

static inline double smat_c_rand_normal_(void) {
    double u1 = smat_c_rand_uniform_();
    double u2 = smat_c_rand_uniform_();
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static inline const char* smat_strerror(smat_err_t err) {
    switch (err) {
        case SMAT_OK: return "Success";
        case SMAT_ERR_NULL: return "Null pointer argument";
        case SMAT_ERR_ALLOC: return "Memory allocation failed";
        case SMAT_ERR_DIM: return "Incompatible matrix dimensions";
        case SMAT_ERR_BOUNDS: return "Matrix index out of bounds";
        case SMAT_ERR_SINGULAR: return "Matrix is singular";
        case SMAT_ERR_NOT_SYMMETRIC: return "Matrix is not symmetric";
        case SMAT_ERR_NOT_POS_DEF: return "Matrix is not symmetric positive-definite";
        case SMAT_ERR_IO: return "File I/O error";
        case SMAT_ERR_INVALID_ARG: return "Invalid argument";
        default: return "Unknown error";
    }
}

/* ----------------------------------------------------------------------------
 * C API: Creation, Destruction, and Memory Management
 * ---------------------------------------------------------------------------- */

static inline void smat_free(smat_t *A) {
    if (!A) return;
    if (A->v) free(A->v);
    if (A->data) free(A->data);
    A->v = NULL;
    A->data = NULL;
    A->rows = 0;
    A->cols = 0;
    free(A);
}

static inline smat_t* smat_create(int rows, int cols) {
    if (rows < 1 || cols < 1) return NULL;
    smat_t *A = (smat_t*)malloc(sizeof(smat_t));
    if (!A) return NULL;
    A->rows = rows;
    A->cols = cols;
    A->data = (double*)calloc((size_t)rows * (size_t)cols, sizeof(double));
    A->v = (double**)malloc((size_t)rows * sizeof(double*));
    if (!A->data || !A->v) {
        smat_free(A);
        return NULL;
    }
    for (int i = 0; i < rows; ++i) {
        A->v[i] = &A->data[(size_t)i * (size_t)cols];
    }
    return A;
}

static inline smat_t* smat_create_val(int rows, int cols, double val) {
    smat_t *A = smat_create(rows, cols);
    if (!A) return NULL;
    size_t total = (size_t)rows * (size_t)cols;
    for (size_t i = 0; i < total; ++i) {
        A->data[i] = val;
    }
    return A;
}

static inline smat_t* smat_zeros(int rows, int cols) {
    return smat_create(rows, cols);
}

static inline smat_t* smat_ones(int rows, int cols) {
    return smat_create_val(rows, cols, 1.0);
}

static inline smat_t* smat_eye(int rows, int cols) {
    smat_t *A = smat_create(rows, cols);
    if (!A) return NULL;
    int k = (rows < cols) ? rows : cols;
    for (int i = 0; i < k; ++i) {
        A->v[i][i] = 1.0;
    }
    return A;
}

static inline smat_t* smat_identity(int n) {
    return smat_eye(n, n);
}

static inline smat_t* smat_rand(int rows, int cols) {
    smat_t *A = smat_create(rows, cols);
    if (!A) return NULL;
    size_t total = (size_t)rows * (size_t)cols;
    for (size_t i = 0; i < total; ++i) {
        A->data[i] = smat_c_rand_uniform_();
    }
    return A;
}

static inline smat_t* smat_randn(int rows, int cols) {
    smat_t *A = smat_create(rows, cols);
    if (!A) return NULL;
    size_t total = (size_t)rows * (size_t)cols;
    for (size_t i = 0; i < total; ++i) {
        A->data[i] = smat_c_rand_normal_();
    }
    return A;
}

static inline smat_t* smat_randperm(int rows, int cols) {
    smat_t *A = smat_create(rows, cols);
    if (!A) return NULL;
    int *perm = (int*)malloc((size_t)rows * sizeof(int));
    if (!perm) {
        smat_free(A);
        return NULL;
    }
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) perm[i] = i + 1;
        for (int i = rows - 1; i > 0; --i) {
            int idx = (int)(smat_c_rand_u64_() % (unsigned long long)(i + 1));
            int tmp = perm[i];
            perm[i] = perm[idx];
            perm[idx] = tmp;
        }
        for (int i = 0; i < rows; ++i) A->v[i][j] = (double)perm[i];
    }
    free(perm);
    return A;
}

static inline smat_t* smat_from_array(int rows, int cols, const double *arr) {
    if (!arr) return NULL;
    smat_t *A = smat_create(rows, cols);
    if (!A) return NULL;
    memcpy(A->data, arr, (size_t)rows * (size_t)cols * sizeof(double));
    return A;
}

static inline smat_t* smat_copy(const smat_t *A) {
    if (!A) return NULL;
    return smat_from_array(A->rows, A->cols, A->data);
}

static inline smat_t* smat_linspace(double start, double end, int n) {
    if (n < 1) return NULL;
    smat_t *A = smat_create(1, n);
    if (!A) return NULL;
    if (n == 1) {
        A->data[0] = end;
        return A;
    }
    double step = (end - start) / (double)(n - 1);
    for (int i = 0; i < n; ++i) {
        A->data[i] = start + i * step;
    }
    A->data[n - 1] = end;
    return A;
}

static inline smat_t* smat_arange(double start, double stop, double step) {
    if (step == 0.0) return NULL;
    if ((stop > start && step < 0.0) || (stop < start && step > 0.0)) return NULL;
    int n = (int)ceil((stop - start) / step - 1e-12);
    if (n < 1) return NULL;
    smat_t *A = smat_create(1, n);
    if (!A) return NULL;
    for (int i = 0; i < n; ++i) {
        A->data[i] = start + i * step;
    }
    return A;
}

static inline smat_t* smat_diag(const smat_t *A) {
    if (!A) return NULL;
    /* If A is a row or column vector (and not 1x1 matrix if treated as vector -> 1x1 diag),
       wait: if A is 1xN or Nx1 (with N > 1), create NxN diagonal matrix.
       Otherwise extract diagonal as min(rows, cols) x 1 column vector. */
    if ((A->rows == 1 && A->cols > 1) || (A->cols == 1 && A->rows > 1)) {
        int n = (A->rows > A->cols) ? A->rows : A->cols;
        smat_t *D = smat_create(n, n);
        if (!D) return NULL;
        for (int i = 0; i < n; ++i) {
            D->v[i][i] = A->data[i];
        }
        return D;
    } else {
        int k = (A->rows < A->cols) ? A->rows : A->cols;
        smat_t *d = smat_create(k, 1);
        if (!d) return NULL;
        for (int i = 0; i < k; ++i) {
            d->v[i][0] = A->v[i][i];
        }
        return d;
    }
}

/* ----------------------------------------------------------------------------
 * C API: Accessors, Properties, and I/O
 * ---------------------------------------------------------------------------- */

static inline int smat_rows(const smat_t *A) { return A ? A->rows : 0; }
static inline int smat_cols(const smat_t *A) { return A ? A->cols : 0; }
static inline int smat_size(const smat_t *A) { return A ? (A->rows * A->cols) : 0; }

static inline double smat_get(const smat_t *A, int r, int c) {
    if (!A || r < 0 || r >= A->rows || c < 0 || c >= A->cols) return 0.0;
    return A->v[r][c];
}

static inline smat_err_t smat_set(smat_t *A, int r, int c, double val) {
    if (!A) return SMAT_ERR_NULL;
    if (r < 0 || r >= A->rows || c < 0 || c >= A->cols) return SMAT_ERR_BOUNDS;
    A->v[r][c] = val;
    return SMAT_OK;
}

static inline smat_err_t smat_fill(smat_t *A, double val) {
    if (!A) return SMAT_ERR_NULL;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) A->data[i] = val;
    return SMAT_OK;
}

static inline bool smat_is_square(const smat_t *A) {
    return A && (A->rows == A->cols) && (A->rows > 0);
}

static inline bool smat_is_symmetric(const smat_t *A, double tol) {
    if (!smat_is_square(A)) return false;
    if (tol < 0.0) tol = 1e-9;
    for (int i = 0; i < A->rows; ++i) {
        for (int j = i + 1; j < A->cols; ++j) {
            if (fabs(A->v[i][j] - A->v[j][i]) > tol) return false;
        }
    }
    return true;
}

static inline bool smat_equals(const smat_t *A, const smat_t *B, double tol) {
    if (!A || !B) return false;
    if (A->rows != B->rows || A->cols != B->cols) return false;
    if (tol < 0.0) tol = 1e-9;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) {
        if (fabs(A->data[i] - B->data[i]) > tol) return false;
    }
    return true;
}

static inline void smat_print(const smat_t *A) {
    if (!A) {
        printf("(null matrix)\n");
        return;
    }
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            if (fabs(A->v[i][j]) < EPSILON)
                printf("%10.4f ", 0.0);
            else
                printf("%10.4f ", A->v[i][j]);
        }
        printf("\n");
    }
}

static inline smat_err_t smat_save_txt(const smat_t *A, const char *filename) {
    if (!A || !filename) return SMAT_ERR_NULL;
    FILE *pFile = fopen(filename, "w");
    if (!pFile) return SMAT_ERR_IO;
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            fprintf(pFile, "%f ", A->v[i][j]);
        }
        fprintf(pFile, "\n");
    }
    fclose(pFile);
    return SMAT_OK;
}

static inline smat_t* smat_load_txt(const char *filename) {
    if (!filename) return NULL;
    FILE *pFile = fopen(filename, "r");
    if (!pFile) return NULL;

    char *line = (char*)malloc(MAX_LINE_LENGTH);
    if (!line) {
        fclose(pFile);
        return NULL;
    }

    int rows = 0;
    int cols = 0;
    while (fgets(line, MAX_LINE_LENGTH, pFile) != NULL) {
        int current_cols = 0;
        char *token = strtok(line, " ,\t\r\n");
        while (token != NULL) {
            current_cols++;
            token = strtok(NULL, " ,\t\r\n");
        }
        if (current_cols == 0) continue;
        if (rows == 0) {
            cols = current_cols;
            rows = 1;
        } else if (current_cols >= cols) {
            rows++;
        } else {
            break;
        }
    }
    fclose(pFile);

    if (rows < 1 || cols < 1) {
        free(line);
        return NULL;
    }

    smat_t *A = smat_create(rows, cols);
    if (!A) {
        free(line);
        return NULL;
    }

    pFile = fopen(filename, "r");
    if (!pFile) {
        free(line);
        smat_free(A);
        return NULL;
    }

    int r = 0;
    while (r < rows && fgets(line, MAX_LINE_LENGTH, pFile) != NULL) {
        char *token = strtok(line, " ,\t\r\n");
        if (!token) continue;
        for (int c = 0; c < cols; ++c) {
            if (c > 0) token = strtok(NULL, " ,\t\r\n");
            A->v[r][c] = token ? atof(token) : 0.0;
        }
        r++;
    }
    fclose(pFile);
    free(line);
    return A;
}

/* ----------------------------------------------------------------------------
 * C API: Structural & Slicing Operations
 * ---------------------------------------------------------------------------- */

static inline smat_t* smat_transpose(const smat_t *A) {
    if (!A) return NULL;
    smat_t *B = smat_create(A->cols, A->rows);
    if (!B) return NULL;
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            B->v[j][i] = A->v[i][j];
        }
    }
    return B;
}

static inline smat_t* smat_reshape(const smat_t *A, int new_rows, int new_cols) {
    if (!A || new_rows < 1 || new_cols < 1) return NULL;
    if (A->rows * A->cols != new_rows * new_cols) return NULL;
    return smat_from_array(new_rows, new_cols, A->data);
}

static inline smat_t* smat_sub(const smat_t *A, int r1, int r2, int c1, int c2) {
    if (!A) return NULL;
    if (r1 < 0 || r2 >= A->rows || c1 < 0 || c2 >= A->cols || r1 > r2 || c1 > c2) return NULL;
    smat_t *B = smat_create(r2 - r1 + 1, c2 - c1 + 1);
    if (!B) return NULL;
    for (int i = r1; i <= r2; ++i) {
        for (int j = c1; j <= c2; ++j) {
            B->v[i - r1][j - c1] = A->v[i][j];
        }
    }
    return B;
}

static inline smat_t* smat_row(const smat_t *A, int r) {
    if (!A || r < 0 || r >= A->rows) return NULL;
    return smat_sub(A, r, r, 0, A->cols - 1);
}

static inline smat_t* smat_col(const smat_t *A, int c) {
    if (!A || c < 0 || c >= A->cols) return NULL;
    return smat_sub(A, 0, A->rows - 1, c, c);
}

static inline smat_err_t smat_set_row(smat_t *A, int r, const smat_t *row_vec) {
    if (!A || !row_vec) return SMAT_ERR_NULL;
    if (r < 0 || r >= A->rows) return SMAT_ERR_BOUNDS;
    if (row_vec->rows * row_vec->cols != A->cols) return SMAT_ERR_DIM;
    for (int j = 0; j < A->cols; ++j) A->v[r][j] = row_vec->data[j];
    return SMAT_OK;
}

static inline smat_err_t smat_set_col(smat_t *A, int c, const smat_t *col_vec) {
    if (!A || !col_vec) return SMAT_ERR_NULL;
    if (c < 0 || c >= A->cols) return SMAT_ERR_BOUNDS;
    if (col_vec->rows * col_vec->cols != A->rows) return SMAT_ERR_DIM;
    for (int i = 0; i < A->rows; ++i) A->v[i][c] = col_vec->data[i];
    return SMAT_OK;
}

static inline smat_err_t smat_swap_rows(smat_t *A, int r1, int r2) {
    if (!A) return SMAT_ERR_NULL;
    if (r1 < 0 || r1 >= A->rows || r2 < 0 || r2 >= A->rows) return SMAT_ERR_BOUNDS;
    if (r1 == r2) return SMAT_OK;
    for (int j = 0; j < A->cols; ++j) {
        double tmp = A->v[r1][j];
        A->v[r1][j] = A->v[r2][j];
        A->v[r2][j] = tmp;
    }
    return SMAT_OK;
}

static inline smat_err_t smat_swap_cols(smat_t *A, int c1, int c2) {
    if (!A) return SMAT_ERR_NULL;
    if (c1 < 0 || c1 >= A->cols || c2 < 0 || c2 >= A->cols) return SMAT_ERR_BOUNDS;
    if (c1 == c2) return SMAT_OK;
    for (int i = 0; i < A->rows; ++i) {
        double tmp = A->v[i][c1];
        A->v[i][c1] = A->v[i][c2];
        A->v[i][c2] = tmp;
    }
    return SMAT_OK;
}

static inline smat_t* smat_concat_right(const smat_t *A, const smat_t *B) {
    if (!A || !B || A->rows != B->rows) return NULL;
    smat_t *C = smat_create(A->rows, A->cols + B->cols);
    if (!C) return NULL;
    for (int i = 0; i < A->rows; ++i) {
        memcpy(&C->v[i][0], A->v[i], (size_t)A->cols * sizeof(double));
        memcpy(&C->v[i][A->cols], B->v[i], (size_t)B->cols * sizeof(double));
    }
    return C;
}

static inline smat_t* smat_concat_bottom(const smat_t *A, const smat_t *B) {
    if (!A || !B || A->cols != B->cols) return NULL;
    smat_t *C = smat_create(A->rows + B->rows, A->cols);
    if (!C) return NULL;
    memcpy(C->data, A->data, (size_t)A->rows * (size_t)A->cols * sizeof(double));
    memcpy(&C->data[(size_t)A->rows * (size_t)A->cols], B->data, (size_t)B->rows * (size_t)B->cols * sizeof(double));
    return C;
}

static inline smat_t* smat_triu(const smat_t *A, int k) {
    if (!A) return NULL;
    smat_t *B = smat_create(A->rows, A->cols);
    if (!B) return NULL;
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            if (j - i >= k) B->v[i][j] = A->v[i][j];
        }
    }
    return B;
}

static inline smat_t* smat_tril(const smat_t *A, int k) {
    if (!A) return NULL;
    smat_t *B = smat_create(A->rows, A->cols);
    if (!B) return NULL;
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            if (j - i <= k) B->v[i][j] = A->v[i][j];
        }
    }
    return B;
}

/* ----------------------------------------------------------------------------
 * C API: Scalar, Element-wise, and Matrix Arithmetic
 * ---------------------------------------------------------------------------- */

static inline smat_err_t smat_add_scalar_self(smat_t *A, double val) {
    if (!A) return SMAT_ERR_NULL;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) A->data[i] += val;
    return SMAT_OK;
}

static inline smat_err_t smat_sub_scalar_self(smat_t *A, double val) {
    return smat_add_scalar_self(A, -val);
}

static inline smat_err_t smat_mul_scalar_self(smat_t *A, double val) {
    if (!A) return SMAT_ERR_NULL;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) A->data[i] *= val;
    return SMAT_OK;
}

static inline smat_err_t smat_div_scalar_self(smat_t *A, double val) {
    if (!A) return SMAT_ERR_NULL;
    if (val == 0.0) return SMAT_ERR_INVALID_ARG;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) A->data[i] /= val;
    return SMAT_OK;
}

static inline smat_t* smat_add_scalar(const smat_t *A, double val) {
    smat_t *B = smat_copy(A);
    if (!B) return NULL;
    smat_add_scalar_self(B, val);
    return B;
}

static inline smat_t* smat_mul_scalar(const smat_t *A, double val) {
    smat_t *B = smat_copy(A);
    if (!B) return NULL;
    smat_mul_scalar_self(B, val);
    return B;
}

static inline smat_t* smat_abs(const smat_t *A) {
    if (!A) return NULL;
    smat_t *B = smat_create(A->rows, A->cols);
    if (!B) return NULL;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) B->data[i] = fabs(A->data[i]);
    return B;
}

static inline smat_t* smat_neg(const smat_t *A) {
    return smat_mul_scalar(A, -1.0);
}

static inline smat_t* smat_exp(const smat_t *A) {
    if (!A) return NULL;
    smat_t *B = smat_create(A->rows, A->cols);
    if (!B) return NULL;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) B->data[i] = exp(A->data[i]);
    return B;
}

static inline smat_t* smat_log(const smat_t *A) {
    if (!A) return NULL;
    smat_t *B = smat_create(A->rows, A->cols);
    if (!B) return NULL;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) B->data[i] = log(A->data[i]);
    return B;
}

static inline smat_t* smat_sqrt(const smat_t *A) {
    if (!A) return NULL;
    smat_t *B = smat_create(A->rows, A->cols);
    if (!B) return NULL;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) B->data[i] = sqrt(A->data[i]);
    return B;
}

static inline smat_t* smat_pow(const smat_t *A, double p) {
    if (!A) return NULL;
    smat_t *B = smat_create(A->rows, A->cols);
    if (!B) return NULL;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) B->data[i] = pow(A->data[i], p);
    return B;
}

static inline smat_t* smat_apply(const smat_t *A, double (*fn)(double)) {
    if (!A || !fn) return NULL;
    smat_t *B = smat_create(A->rows, A->cols);
    if (!B) return NULL;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) B->data[i] = fn(A->data[i]);
    return B;
}

static inline smat_err_t smat_add_self(smat_t *A, const smat_t *B) {
    if (!A || !B) return SMAT_ERR_NULL;
    if (A->rows != B->rows || A->cols != B->cols) return SMAT_ERR_DIM;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) A->data[i] += B->data[i];
    return SMAT_OK;
}

static inline smat_err_t smat_sub_self(smat_t *A, const smat_t *B) {
    if (!A || !B) return SMAT_ERR_NULL;
    if (A->rows != B->rows || A->cols != B->cols) return SMAT_ERR_DIM;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) A->data[i] -= B->data[i];
    return SMAT_OK;
}

static inline smat_err_t smat_dot_mul_self(smat_t *A, const smat_t *B) {
    if (!A || !B) return SMAT_ERR_NULL;
    if (A->rows != B->rows || A->cols != B->cols) return SMAT_ERR_DIM;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) A->data[i] *= B->data[i];
    return SMAT_OK;
}

static inline smat_err_t smat_dot_div_self(smat_t *A, const smat_t *B) {
    if (!A || !B) return SMAT_ERR_NULL;
    if (A->rows != B->rows || A->cols != B->cols) return SMAT_ERR_DIM;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) A->data[i] /= B->data[i];
    return SMAT_OK;
}

static inline smat_t* smat_add(const smat_t *A, const smat_t *B) {
    if (!A || !B || A->rows != B->rows || A->cols != B->cols) return NULL;
    smat_t *C = smat_copy(A);
    if (!C) return NULL;
    smat_add_self(C, B);
    return C;
}

static inline smat_t* smat_sub_mat(const smat_t *A, const smat_t *B) {
    if (!A || !B || A->rows != B->rows || A->cols != B->cols) return NULL;
    smat_t *C = smat_copy(A);
    if (!C) return NULL;
    smat_sub_self(C, B);
    return C;
}

static inline smat_t* smat_dot_mul(const smat_t *A, const smat_t *B) {
    if (!A || !B || A->rows != B->rows || A->cols != B->cols) return NULL;
    smat_t *C = smat_copy(A);
    if (!C) return NULL;
    smat_dot_mul_self(C, B);
    return C;
}

static inline smat_t* smat_dot_div(const smat_t *A, const smat_t *B) {
    if (!A || !B || A->rows != B->rows || A->cols != B->cols) return NULL;
    smat_t *C = smat_copy(A);
    if (!C) return NULL;
    smat_dot_div_self(C, B);
    return C;
}

/* Cache-friendly i-k-j matrix multiplication */
static inline smat_t* smat_mul(const smat_t *A, const smat_t *B) {
    if (!A || !B || A->cols != B->rows) return NULL;
    smat_t *C = smat_create(A->rows, B->cols);
    if (!C) return NULL;
    for (int i = 0; i < A->rows; ++i) {
        double *c_row = C->v[i];
        const double *a_row = A->v[i];
        for (int k = 0; k < A->cols; ++k) {
            double a_ik = a_row[k];
            const double *b_row = B->v[k];
            for (int j = 0; j < B->cols; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
    return C;
}

static inline double smat_dot(const smat_t *A, const smat_t *B) {
    if (!A || !B || A->rows * A->cols != B->rows * B->cols) return 0.0;
    double s = 0.0;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) s += A->data[i] * B->data[i];
    return s;
}

static inline smat_t* smat_outer(const smat_t *a, const smat_t *b) {
    if (!a || !b) return NULL;
    int m = a->rows * a->cols;
    int n = b->rows * b->cols;
    smat_t *C = smat_create(m, n);
    if (!C) return NULL;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C->v[i][j] = a->data[i] * b->data[j];
        }
    }
    return C;
}

static inline smat_t* smat_cross(const smat_t *a, const smat_t *b) {
    if (!a || !b) return NULL;
    if (a->rows * a->cols != 3 || b->rows * b->cols != 3) return NULL;
    smat_t *c = smat_create(a->rows, a->cols);
    if (!c) return NULL;
    c->data[0] = a->data[1] * b->data[2] - a->data[2] * b->data[1];
    c->data[1] = a->data[2] * b->data[0] - a->data[0] * b->data[2];
    c->data[2] = a->data[0] * b->data[1] - a->data[1] * b->data[0];
    return c;
}

static inline smat_t* smat_kron(const smat_t *A, const smat_t *B) {
    if (!A || !B) return NULL;
    smat_t *K = smat_create(A->rows * B->rows, A->cols * B->cols);
    if (!K) return NULL;
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            double a_ij = A->v[i][j];
            for (int r = 0; r < B->rows; ++r) {
                for (int c = 0; c < B->cols; ++c) {
                    K->v[i * B->rows + r][j * B->cols + c] = a_ij * B->v[r][c];
                }
            }
        }
    }
    return K;
}

/* ----------------------------------------------------------------------------
 * C API: Reductions, Norms, and Statistics
 * ---------------------------------------------------------------------------- */

static inline double smat_sum(const smat_t *A) {
    if (!A) return 0.0;
    double s = 0.0;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) s += A->data[i];
    return s;
}

static inline double smat_mean(const smat_t *A) {
    if (!A || A->rows * A->cols == 0) return 0.0;
    return smat_sum(A) / (double)(A->rows * A->cols);
}

static inline double smat_var(const smat_t *A, bool sample) {
    if (!A) return 0.0;
    int n = A->rows * A->cols;
    if (n <= (sample ? 1 : 0)) return 0.0;
    double m = smat_mean(A);
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = A->data[i] - m;
        s += d * d;
    }
    return s / (double)(sample ? (n - 1) : n);
}

static inline double smat_std(const smat_t *A) {
    return sqrt(smat_var(A, false));
}

static inline double smat_std_sample(const smat_t *A) {
    return sqrt(smat_var(A, true));
}

static inline double smat_trace(const smat_t *A) {
    if (!A) return 0.0;
    int k = (A->rows < A->cols) ? A->rows : A->cols;
    double s = 0.0;
    for (int i = 0; i < k; ++i) s += A->v[i][i];
    return s;
}

static inline double smat_fnorm(const smat_t *A) {
    if (!A) return 0.0;
    double s = 0.0;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) s += A->data[i] * A->data[i];
    return sqrt(s);
}

static inline double smat_pnorm(const smat_t *A, double p) {
    if (!A || p <= 0.0) return 0.0;
    double s = 0.0;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t i = 0; i < total; ++i) s += pow(fabs(A->data[i]), p);
    return pow(s, 1.0 / p);
}

static inline double smat_norm_1(const smat_t *A) {
    if (!A) return 0.0;
    double max_col = 0.0;
    for (int j = 0; j < A->cols; ++j) {
        double col_sum = 0.0;
        for (int i = 0; i < A->rows; ++i) col_sum += fabs(A->v[i][j]);
        if (col_sum > max_col) max_col = col_sum;
    }
    return max_col;
}

static inline double smat_norm_inf(const smat_t *A) {
    if (!A) return 0.0;
    double max_row = 0.0;
    for (int i = 0; i < A->rows; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < A->cols; ++j) row_sum += fabs(A->v[i][j]);
        if (row_sum > max_row) max_row = row_sum;
    }
    return max_row;
}

static inline double smat_max(const smat_t *A, int *r_out, int *c_out) {
    if (!A || A->rows < 1 || A->cols < 1) return 0.0;
    double max_v = A->v[0][0];
    int mr = 0, mc = 0;
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            if (A->v[i][j] > max_v) {
                max_v = A->v[i][j];
                mr = i;
                mc = j;
            }
        }
    }
    if (r_out) *r_out = mr;
    if (c_out) *c_out = mc;
    return max_v;
}

static inline double smat_min(const smat_t *A, int *r_out, int *c_out) {
    if (!A || A->rows < 1 || A->cols < 1) return 0.0;
    double min_v = A->v[0][0];
    int mr = 0, mc = 0;
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            if (A->v[i][j] < min_v) {
                min_v = A->v[i][j];
                mr = i;
                mc = j;
            }
        }
    }
    if (r_out) *r_out = mr;
    if (c_out) *c_out = mc;
    return min_v;
}

static inline smat_t* smat_sum_axis(const smat_t *A, int axis) {
    if (!A) return NULL;
    if (axis == 0) {
        smat_t *S = smat_create(1, A->cols);
        if (!S) return NULL;
        for (int i = 0; i < A->rows; ++i) {
            for (int j = 0; j < A->cols; ++j) S->v[0][j] += A->v[i][j];
        }
        return S;
    } else if (axis == 1) {
        smat_t *S = smat_create(A->rows, 1);
        if (!S) return NULL;
        for (int i = 0; i < A->rows; ++i) {
            for (int j = 0; j < A->cols; ++j) S->v[i][0] += A->v[i][j];
        }
        return S;
    }
    return NULL;
}

static inline smat_t* smat_mean_axis(const smat_t *A, int axis) {
    smat_t *S = smat_sum_axis(A, axis);
    if (!S) return NULL;
    double denom = (axis == 0) ? (double)A->rows : (double)A->cols;
    smat_div_scalar_self(S, denom);
    return S;
}

/* Sample covariance matrix of data matrix X (rows = observations, cols = variables) */
static inline smat_t* smat_cov(const smat_t *X) {
    if (!X || X->rows < 1) return NULL;
    int n = X->rows;
    int p = X->cols;
    smat_t *Xc = smat_copy(X);
    if (!Xc) return NULL;
    for (int j = 0; j < p; ++j) {
        double m = 0.0;
        for (int i = 0; i < n; ++i) m += Xc->v[i][j];
        m /= (double)n;
        for (int i = 0; i < n; ++i) Xc->v[i][j] -= m;
    }
    smat_t *Xt = smat_transpose(Xc);
    smat_t *C = smat_mul(Xt, Xc);
    smat_free(Xc);
    smat_free(Xt);
    if (!C) return NULL;
    double denom = (n > 1) ? (double)(n - 1) : 1.0;
    smat_div_scalar_self(C, denom);
    return C;
}

static inline smat_t* smat_corrcoef(const smat_t *X) {
    smat_t *C = smat_cov(X);
    if (!C) return NULL;
    int p = C->rows;
    smat_t *R = smat_create(p, p);
    if (!R) {
        smat_free(C);
        return NULL;
    }
    for (int i = 0; i < p; ++i) {
        R->v[i][i] = 1.0;
        for (int j = i + 1; j < p; ++j) {
            double denom = sqrt(C->v[i][i] * C->v[j][j]);
            double val = (denom > 1e-15) ? (C->v[i][j] / denom) : 0.0;
            if (val > 1.0) val = 1.0;
            if (val < -1.0) val = -1.0;
            R->v[i][j] = val;
            R->v[j][i] = val;
        }
    }
    smat_free(C);
    return R;
}

/* ----------------------------------------------------------------------------
 * C API: Linear Algebra Decompositions & Solvers
 * ---------------------------------------------------------------------------- */

static inline smat_err_t smat_lu(const smat_t *A, smat_t **L_out, smat_t **U_out, smat_t **P_out) {
    if (!A || !L_out || !U_out || !P_out) return SMAT_ERR_NULL;
    if (A->rows != A->cols) return SMAT_ERR_DIM;
    int n = A->rows;
    smat_t *L = smat_zeros(n, n);
    smat_t *U = smat_zeros(n, n);
    smat_t *P = smat_identity(n);
    smat_t *W = smat_copy(A);
    if (!L || !U || !P || !W) {
        smat_free(L); smat_free(U); smat_free(P); smat_free(W);
        return SMAT_ERR_ALLOC;
    }

    for (int i = 0; i < n; ++i) {
        double max_el = fabs(W->v[i][i]);
        int max_row = i;
        for (int k = i + 1; k < n; ++k) {
            double val = fabs(W->v[k][i]);
            if (val > max_el) {
                max_el = val;
                max_row = k;
            }
        }

        if (max_row != i) {
            smat_swap_rows(W, i, max_row);
            smat_swap_rows(P, i, max_row);
            for (int k = 0; k < i; ++k) {
                double tmp = L->v[i][k];
                L->v[i][k] = L->v[max_row][k];
                L->v[max_row][k] = tmp;
            }
        }

        L->v[i][i] = 1.0;
        if (fabs(W->v[i][i]) > 1e-15) {
            for (int k = i + 1; k < n; ++k) {
                double mult = W->v[k][i] / W->v[i][i];
                L->v[k][i] = mult;
                W->v[k][i] = 0.0;
                for (int j = i + 1; j < n; ++j) {
                    W->v[k][j] -= mult * W->v[i][j];
                }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            U->v[i][j] = W->v[i][j];
        }
    }

    smat_free(W);
    *L_out = L;
    *U_out = U;
    *P_out = P;
    return SMAT_OK;
}

static inline smat_t* smat_cholesky(const smat_t *A) {
    if (!A || A->rows != A->cols) return NULL;
    if (!smat_is_symmetric(A, 1e-7)) return NULL;
    int n = A->rows;
    smat_t *L = smat_zeros(n, n);
    if (!L) return NULL;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A->v[i][j];
            for (int k = 0; k < j; ++k) {
                sum -= L->v[i][k] * L->v[j][k];
            }
            if (i == j) {
                if (sum <= 1e-15) {
                    smat_free(L);
                    return NULL;
                }
                L->v[i][i] = sqrt(sum);
            } else {
                L->v[i][j] = sum / L->v[j][j];
            }
        }
    }
    return L;
}

static inline smat_err_t smat_qr(const smat_t *A, smat_t **Q_out, smat_t **R_out) {
    if (!A || !Q_out || !R_out) return SMAT_ERR_NULL;
    int m = A->rows;
    int n = A->cols;
    smat_t *Q = smat_identity(m);
    smat_t *R = smat_copy(A);
    double *v_vec = (double*)calloc((size_t)m, sizeof(double));
    if (!Q || !R || !v_vec) {
        smat_free(Q); smat_free(R); free(v_vec);
        return SMAT_ERR_ALLOC;
    }

    int steps = (m - 1 < n) ? (m - 1) : n;
    for (int k = 0; k < steps; ++k) {
        double norm_x = 0.0;
        for (int i = k; i < m; ++i) norm_x += R->v[i][k] * R->v[i][k];
        norm_x = sqrt(norm_x);
        if (norm_x < 1e-15) continue;

        double u1 = R->v[k][k];
        double s = (u1 >= 0.0) ? 1.0 : -1.0;
        v_vec[k] = u1 + s * norm_x;
        for (int i = k + 1; i < m; ++i) v_vec[i] = R->v[i][k];

        double norm_v_sq = 0.0;
        for (int i = k; i < m; ++i) norm_v_sq += v_vec[i] * v_vec[i];
        if (norm_v_sq < 1e-30) continue;
        double tau = 2.0 / norm_v_sq;

        for (int j = k; j < n; ++j) {
            double dot = 0.0;
            for (int i = k; i < m; ++i) dot += v_vec[i] * R->v[i][j];
            for (int i = k; i < m; ++i) R->v[i][j] -= tau * dot * v_vec[i];
        }

        for (int i = 0; i < m; ++i) {
            double dot = 0.0;
            for (int j = k; j < m; ++j) dot += Q->v[i][j] * v_vec[j];
            for (int j = k; j < m; ++j) Q->v[i][j] -= tau * dot * v_vec[j];
        }
    }

    int diag_len = (m < n) ? m : n;
    for (int k = 0; k < diag_len; ++k) {
        for (int i = k + 1; i < m; ++i) R->v[i][k] = 0.0;
        if (R->v[k][k] < 0.0) {
            for (int j = k; j < n; ++j) R->v[k][j] = -R->v[k][j];
            for (int i = 0; i < m; ++i) Q->v[i][k] = -Q->v[i][k];
        }
    }

    free(v_vec);
    *Q_out = Q;
    *R_out = R;
    return SMAT_OK;
}

static inline smat_err_t smat_eigen(const smat_t *A, smat_t **V_out, smat_t **D_out) {
    if (!A || !V_out || !D_out) return SMAT_ERR_NULL;
    if (A->rows != A->cols) return SMAT_ERR_DIM;
    if (!smat_is_symmetric(A, 1e-7)) return SMAT_ERR_NOT_SYMMETRIC;

    int n = A->rows;
    smat_t *D = smat_copy(A);
    smat_t *V = smat_identity(n);
    if (!D || !V) {
        smat_free(D); smat_free(V);
        return SMAT_ERR_ALLOC;
    }

    double tol = SMAT_JACOBI_EPS * (1.0 + smat_fnorm(A));
    int max_rotations = 50 * n * n + 100;

    for (int iter = 0; iter < max_rotations; ++iter) {
        double max_el = 0.0;
        int p = -1, q = -1;
        for (int r = 0; r < n - 1; ++r) {
            for (int c = r + 1; c < n; ++c) {
                double val = fabs(D->v[r][c]);
                if (val > max_el) {
                    max_el = val;
                    p = r;
                    q = c;
                }
            }
        }
        if (max_el < tol || p < 0) break;

        double App = D->v[p][p];
        double Aqq = D->v[q][q];
        double Apq = D->v[p][q];

        double theta = 0.5 * atan2(2.0 * Apq, Aqq - App);
        double c = cos(theta);
        double s = sin(theta);

        D->v[p][p] = c * c * App - 2.0 * s * c * Apq + s * s * Aqq;
        D->v[q][q] = s * s * App + 2.0 * s * c * Apq + c * c * Aqq;
        D->v[p][q] = 0.0;
        D->v[q][p] = 0.0;

        for (int k = 0; k < n; ++k) {
            if (k != p && k != q) {
                double Dpk = D->v[p][k];
                double Dqk = D->v[q][k];
                D->v[p][k] = c * Dpk - s * Dqk;
                D->v[k][p] = D->v[p][k];
                D->v[q][k] = s * Dpk + c * Dqk;
                D->v[k][q] = D->v[q][k];
            }
        }

        for (int k = 0; k < n; ++k) {
            double Vkp = V->v[k][p];
            double Vkq = V->v[k][q];
            V->v[k][p] = c * Vkp - s * Vkq;
            V->v[k][q] = s * Vkp + c * Vkq;
        }
    }

    /* Clean off-diagonals and sort eigenvalues in descending order */
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) D->v[i][j] = 0.0;
        }
    }
    for (int i = 0; i < n - 1; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < n; ++j) {
            if (D->v[j][j] > D->v[max_idx][max_idx]) max_idx = j;
        }
        if (max_idx != i) {
            double tmp = D->v[i][i];
            D->v[i][i] = D->v[max_idx][max_idx];
            D->v[max_idx][max_idx] = tmp;
            smat_swap_cols(V, i, max_idx);
        }
    }

    *V_out = V;
    *D_out = D;
    return SMAT_OK;
}

static inline smat_t* smat_eigenvalues(const smat_t *A) {
    smat_t *V = NULL, *D = NULL;
    if (smat_eigen(A, &V, &D) != SMAT_OK) return NULL;
    smat_t *eig = smat_create(A->rows, 1);
    if (eig) {
        for (int i = 0; i < A->rows; ++i) eig->v[i][0] = D->v[i][i];
    }
    smat_free(V);
    smat_free(D);
    return eig;
}

static inline smat_err_t smat_svd(const smat_t *A, smat_t **U_out, smat_t **S_out, smat_t **Vt_out) {
    if (!A || !U_out || !S_out || !Vt_out) return SMAT_ERR_NULL;
    if (A->rows < A->cols) {
        smat_t *At = smat_transpose(A);
        if (!At) return SMAT_ERR_ALLOC;
        smat_t *Ut = NULL, *St = NULL, *Vtt = NULL;
        smat_err_t err = smat_svd(At, &Ut, &St, &Vtt);
        smat_free(At);
        if (err != SMAT_OK) return err;
        *U_out = smat_transpose(Vtt);
        *S_out = smat_transpose(St);
        *Vt_out = smat_transpose(Ut);
        smat_free(Ut); smat_free(St); smat_free(Vtt);
        if (!*U_out || !*S_out || !*Vt_out) {
            smat_free(*U_out); smat_free(*S_out); smat_free(*Vt_out);
            return SMAT_ERR_ALLOC;
        }
        return SMAT_OK;
    }

    int m = A->rows;
    int n = A->cols;
    smat_t *U = smat_copy(A);
    smat_t *V = smat_identity(n);
    smat_t *S = smat_zeros(n, n);
    double *sigmas = (double*)calloc((size_t)n, sizeof(double));
    if (!U || !V || !S || !sigmas) {
        smat_free(U); smat_free(V); smat_free(S); free(sigmas);
        return SMAT_ERR_ALLOC;
    }

    int max_sweeps = 60;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        int count = 0;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double a = 0.0, b = 0.0, c = 0.0;
                for (int k = 0; k < m; ++k) {
                    double p = U->v[k][i];
                    double q = U->v[k][j];
                    a += p * p;
                    b += p * q;
                    c += q * q;
                }
                if (fabs(b) <= SMAT_JACOBI_EPS * sqrt(a * c)) continue;
                count++;

                double zeta = (c - a) / (2.0 * b);
                double t = ((zeta >= 0.0) ? 1.0 : -1.0) / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                double cs = 1.0 / sqrt(1.0 + t * t);
                double sn = cs * t;

                for (int k = 0; k < m; ++k) {
                    double p = U->v[k][i];
                    double q = U->v[k][j];
                    U->v[k][i] = cs * p - sn * q;
                    U->v[k][j] = sn * p + cs * q;
                }
                for (int k = 0; k < n; ++k) {
                    double p = V->v[k][i];
                    double q = V->v[k][j];
                    V->v[k][i] = cs * p - sn * q;
                    V->v[k][j] = sn * p + cs * q;
                }
            }
        }
        if (count == 0) break;
    }

    for (int i = 0; i < n; ++i) {
        double norm = 0.0;
        for (int k = 0; k < m; ++k) norm += U->v[k][i] * U->v[k][i];
        sigmas[i] = sqrt(norm);
    }

    /* Sort singular values and columns of U, V in descending order */
    for (int i = 0; i < n - 1; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < n; ++j) {
            if (sigmas[j] > sigmas[max_idx]) max_idx = j;
        }
        if (max_idx != i) {
            double tmp = sigmas[i];
            sigmas[i] = sigmas[max_idx];
            sigmas[max_idx] = tmp;
            smat_swap_cols(U, i, max_idx);
            smat_swap_cols(V, i, max_idx);
        }
    }

    double sigma_max = (n > 0) ? sigmas[0] : 0.0;
    double tol = (double)((m > n) ? m : n) * ((sigma_max > 1.0) ? sigma_max : 1.0) * 1e-14;

    for (int i = 0; i < n; ++i) {
        if (sigmas[i] > tol) {
            S->v[i][i] = sigmas[i];
            for (int k = 0; k < m; ++k) U->v[k][i] /= sigmas[i];
        } else {
            S->v[i][i] = 0.0;
            /* Complete zero singular value column with a unit vector orthogonal to previous columns */
            for (int cand = 0; cand < m; ++cand) {
                for (int k = 0; k < m; ++k) U->v[k][i] = (k == cand) ? 1.0 : 0.0;
                for (int pass = 0; pass < 2; ++pass) {
                    for (int prev = 0; prev < i; ++prev) {
                        double dot = 0.0;
                        for (int k = 0; k < m; ++k) dot += U->v[k][prev] * U->v[k][i];
                        for (int k = 0; k < m; ++k) U->v[k][i] -= dot * U->v[k][prev];
                    }
                }
                double nrm = 0.0;
                for (int k = 0; k < m; ++k) nrm += U->v[k][i] * U->v[k][i];
                nrm = sqrt(nrm);
                if (nrm > 0.1) {
                    for (int k = 0; k < m; ++k) U->v[k][i] /= nrm;
                    break;
                }
            }
        }
    }

    free(sigmas);
    *Vt_out = smat_transpose(V);
    smat_free(V);
    if (!*Vt_out) {
        smat_free(U); smat_free(S);
        return SMAT_ERR_ALLOC;
    }
    *U_out = U;
    *S_out = S;
    return SMAT_OK;
}

static inline double smat_determinant(const smat_t *A) {
    if (!smat_is_square(A)) return 0.0;
    smat_t *W = smat_copy(A);
    if (!W) return 0.0;
    int n = W->rows;
    double det = 1.0;
    double scale = smat_norm_inf(W);
    if (scale == 0.0) {
        smat_free(W);
        return 0.0;
    }
    double tol = 1e-14 * scale;

    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (fabs(W->v[j][i]) > fabs(W->v[pivot][i])) pivot = j;
        }
        if (fabs(W->v[pivot][i]) <= tol) {
            smat_free(W);
            return 0.0;
        }
        if (pivot != i) {
            smat_swap_rows(W, i, pivot);
            det = -det;
        }
        det *= W->v[i][i];
        for (int j = i + 1; j < n; ++j) {
            double factor = W->v[j][i] / W->v[i][i];
            for (int k = i + 1; k < n; ++k) {
                W->v[j][k] -= factor * W->v[i][k];
            }
        }
    }
    smat_free(W);
    return det;
}

static inline int smat_rank(const smat_t *A) {
    if (!A) return 0;
    smat_t *U = NULL, *S = NULL, *Vt = NULL;
    if (smat_svd(A, &U, &S, &Vt) != SMAT_OK) return 0;
    int k = (A->rows < A->cols) ? A->rows : A->cols;
    int max_dim = (A->rows > A->cols) ? A->rows : A->cols;
    double sigma_max = (k > 0) ? S->v[0][0] : 0.0;
    double tol = (double)max_dim * ((sigma_max > 1.0) ? sigma_max : 1.0) * 1e-12;
    int r = 0;
    for (int i = 0; i < k; ++i) {
        if (S->v[i][i] > tol) r++;
    }
    smat_free(U); smat_free(S); smat_free(Vt);
    return r;
}

static inline double smat_norm_2(const smat_t *A) {
    if (!A) return 0.0;
    smat_t *U = NULL, *S = NULL, *Vt = NULL;
    if (smat_svd(A, &U, &S, &Vt) != SMAT_OK) return 0.0;
    double s_max = S->v[0][0];
    smat_free(U); smat_free(S); smat_free(Vt);
    return s_max;
}

static inline double smat_cond(const smat_t *A) {
    if (!A) return 0.0;
    smat_t *U = NULL, *S = NULL, *Vt = NULL;
    if (smat_svd(A, &U, &S, &Vt) != SMAT_OK) return 0.0;
    int k = (A->rows < A->cols) ? A->rows : A->cols;
    double s_max = S->v[0][0];
    double s_min = S->v[k - 1][k - 1];
    smat_free(U); smat_free(S); smat_free(Vt);
    if (s_min <= 1e-15 * s_max) return 1.0 / 0.0; /* INFINITY */
    return s_max / s_min;
}

static inline smat_t* smat_inverse(const smat_t *A) {
    if (!smat_is_square(A)) return NULL;
    int n = A->rows;
    smat_t *Inv = smat_identity(n);
    smat_t *W = smat_copy(A);
    if (!Inv || !W) {
        smat_free(Inv); smat_free(W);
        return NULL;
    }

    double scale = smat_norm_inf(W);
    double tol = 1e-14 * ((scale > 1.0) ? scale : 1.0);

    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (fabs(W->v[j][i]) > fabs(W->v[pivot][i])) pivot = j;
        }
        if (fabs(W->v[pivot][i]) <= tol) {
            smat_free(Inv); smat_free(W);
            return NULL;
        }
        if (pivot != i) {
            smat_swap_rows(W, i, pivot);
            smat_swap_rows(Inv, i, pivot);
        }
        double div = W->v[i][i];
        for (int k = 0; k < n; ++k) {
            W->v[i][k] /= div;
            Inv->v[i][k] /= div;
        }
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                double mul = W->v[j][i];
                for (int k = 0; k < n; ++k) {
                    W->v[j][k] -= mul * W->v[i][k];
                    Inv->v[j][k] -= mul * Inv->v[i][k];
                }
            }
        }
    }
    smat_free(W);
    return Inv;
}

static inline smat_t* smat_pinv(const smat_t *A) {
    if (!A) return NULL;
    smat_t *U = NULL, *S = NULL, *Vt = NULL;
    if (smat_svd(A, &U, &S, &Vt) != SMAT_OK) return NULL;
    int k = S->rows;
    int max_dim = (A->rows > A->cols) ? A->rows : A->cols;
    double sigma_max = (k > 0) ? S->v[0][0] : 0.0;
    double tol = (double)max_dim * ((sigma_max > 1.0) ? sigma_max : 1.0) * 1e-12;

    smat_t *V = smat_transpose(Vt);
    smat_t *Ut = smat_transpose(U);
    if (!V || !Ut) {
        smat_free(U); smat_free(S); smat_free(Vt); smat_free(V); smat_free(Ut);
        return NULL;
    }
    for (int i = 0; i < k; ++i) {
        double s_inv = (S->v[i][i] > tol) ? (1.0 / S->v[i][i]) : 0.0;
        for (int j = 0; j < V->rows; ++j) {
            V->v[j][i] *= s_inv;
        }
    }
    smat_t *Pinv = smat_mul(V, Ut);
    smat_free(U); smat_free(S); smat_free(Vt); smat_free(V); smat_free(Ut);
    return Pinv;
}

static inline smat_t* smat_solve(const smat_t *A, const smat_t *B) {
    if (!A || !B || A->rows != A->cols || A->rows != B->rows) return NULL;
    int n = A->rows;
    smat_t *L = NULL, *U = NULL, *P = NULL;
    if (smat_lu(A, &L, &U, &P) != SMAT_OK) return NULL;

    double scale = smat_norm_inf(A);
    double tol = 1e-14 * ((scale > 1.0) ? scale : 1.0);
    for (int i = 0; i < n; ++i) {
        if (fabs(U->v[i][i]) <= tol) {
            smat_free(L); smat_free(U); smat_free(P);
            return NULL;
        }
    }

    smat_t *PB = smat_mul(P, B);
    smat_t *Y = smat_create(n, B->cols);
    smat_t *X = smat_create(n, B->cols);
    if (!PB || !Y || !X) {
        smat_free(L); smat_free(U); smat_free(P);
        smat_free(PB); smat_free(Y); smat_free(X);
        return NULL;
    }

    for (int k = 0; k < B->cols; ++k) {
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int j = 0; j < i; ++j) s += L->v[i][j] * Y->v[j][k];
            Y->v[i][k] = PB->v[i][k] - s;
        }
        for (int i = n - 1; i >= 0; --i) {
            double s = 0.0;
            for (int j = i + 1; j < n; ++j) s += U->v[i][j] * X->v[j][k];
            X->v[i][k] = (Y->v[i][k] - s) / U->v[i][i];
        }
    }

    smat_free(L); smat_free(U); smat_free(P);
    smat_free(PB); smat_free(Y);
    return X;
}

static inline smat_t* smat_lstsq(const smat_t *A, const smat_t *B) {
    if (!A || !B || A->rows != B->rows) return NULL;
    smat_t *Pinv = smat_pinv(A);
    if (!Pinv) return NULL;
    smat_t *X = smat_mul(Pinv, B);
    smat_free(Pinv);
    return X;
}

static inline smat_t* smat_orth(const smat_t *A) {
    if (!A) return NULL;
    smat_t *U = NULL, *S = NULL, *Vt = NULL;
    if (smat_svd(A, &U, &S, &Vt) != SMAT_OK) return NULL;
    int k = S->rows;
    int max_dim = (A->rows > A->cols) ? A->rows : A->cols;
    double sigma_max = (k > 0) ? S->v[0][0] : 0.0;
    double tol = (double)max_dim * ((sigma_max > 1.0) ? sigma_max : 1.0) * 1e-12;
    int r = 0;
    for (int i = 0; i < k; ++i) {
        if (S->v[i][i] > tol) r++;
    }
    smat_t *Q = (r > 0) ? smat_sub(U, 0, A->rows - 1, 0, r - 1) : NULL;
    smat_free(U); smat_free(S); smat_free(Vt);
    return Q;
}

static inline smat_t* smat_null_space(const smat_t *A) {
    if (!A) return NULL;
    /* Right singular vectors with zero singular value are eigenvectors of A^T * A with eigenvalue 0 */
    smat_t *At = smat_transpose(A);
    if (!At) return NULL;
    smat_t *AtA = smat_mul(At, A);
    smat_free(At);
    if (!AtA) return NULL;
    smat_t *V = NULL, *D = NULL;
    if (smat_eigen(AtA, &V, &D) != SMAT_OK) {
        smat_free(AtA);
        return NULL;
    }
    smat_free(AtA);
    int n = A->cols;
    double lambda_max = (n > 0 && D->v[0][0] > 0.0) ? D->v[0][0] : 0.0;
    double tol = (double)((A->rows > n) ? A->rows : n) * ((lambda_max > 1.0) ? lambda_max : 1.0) * 1e-11;
    int r = 0;
    for (int i = 0; i < n; ++i) {
        if (D->v[i][i] > tol) r++;
    }
    smat_t *N = (r < n) ? smat_sub(V, 0, n - 1, r, n - 1) : NULL;
    smat_free(V); smat_free(D);
    return N;
}

static inline smat_t* smat_expm(const smat_t *A) {
    if (!smat_is_square(A)) return NULL;
    int n = A->rows;
    double nrm = smat_norm_inf(A);
    int s = 0;
    if (nrm > 0.5) {
        s = (int)ceil(log(nrm) / log(2.0)) + 1;
        if (s < 0) s = 0;
    }
    double scale = pow(2.0, (double)(-s));
    smat_t *B = smat_mul_scalar(A, scale);
    smat_t *B2 = smat_mul(B, B);
    smat_t *B4 = smat_mul(B2, B2);
    smat_t *B6 = smat_mul(B2, B4);
    smat_t *I = smat_identity(n);
    if (!B || !B2 || !B4 || !B6 || !I) {
        smat_free(B); smat_free(B2); smat_free(B4); smat_free(B6); smat_free(I);
        return NULL;
    }

    const double c0 = 1.0;
    const double c1 = 0.5;
    const double c2 = 5.0 / 44.0;
    const double c3 = 1.0 / 66.0;
    const double c4 = 1.0 / 792.0;
    const double c5 = 1.0 / 15840.0;
    const double c6 = 1.0 / 665280.0;

    smat_t *U_inner = smat_zeros(n, n);
    smat_t *V = smat_zeros(n, n);
    size_t total = (size_t)n * (size_t)n;
    for (size_t i = 0; i < total; ++i) {
        U_inner->data[i] = c1 * I->data[i] + c3 * B2->data[i] + c5 * B4->data[i];
        V->data[i] = c0 * I->data[i] + c2 * B2->data[i] + c4 * B4->data[i] + c6 * B6->data[i];
    }
    smat_t *U = smat_mul(B, U_inner);
    smat_t *VmU = smat_sub_mat(V, U);
    smat_t *VpU = smat_add(V, U);
    smat_t *R = smat_solve(VmU, VpU);

    smat_free(B); smat_free(B2); smat_free(B4); smat_free(B6); smat_free(I);
    smat_free(U_inner); smat_free(V); smat_free(U); smat_free(VmU); smat_free(VpU);

    for (int k = 0; k < s && R != NULL; ++k) {
        smat_t *R2 = smat_mul(R, R);
        smat_free(R);
        R = R2;
    }
    return R;
}

/* ----------------------------------------------------------------------------
 * C API: PCA, Pairwise Distances, and Multidimensional Scaling (MDS)
 * ---------------------------------------------------------------------------- */

static inline smat_err_t smat_pca(const smat_t *X, int n_components,
                                  smat_t **projected_out,
                                  smat_t **components_out,
                                  smat_t **explained_variance_out) {
    if (!X || n_components < 1 || n_components > X->cols) return SMAT_ERR_INVALID_ARG;
    int n = X->rows;
    int p = X->cols;
    smat_t *Xc = smat_copy(X);
    if (!Xc) return SMAT_ERR_ALLOC;
    for (int j = 0; j < p; ++j) {
        double m = 0.0;
        for (int i = 0; i < n; ++i) m += Xc->v[i][j];
        m /= (double)n;
        for (int i = 0; i < n; ++i) Xc->v[i][j] -= m;
    }
    smat_t *C = smat_cov(Xc);
    smat_t *V = NULL, *D = NULL;
    if (!C || smat_eigen(C, &V, &D) != SMAT_OK) {
        smat_free(Xc); smat_free(C);
        return SMAT_ERR_ALLOC;
    }
    smat_free(C);

    smat_t *W = smat_sub(V, 0, p - 1, 0, n_components - 1);
    smat_t *proj = smat_mul(Xc, W);
    smat_t *exp_var = smat_create(n_components, 1);
    if (exp_var) {
        for (int i = 0; i < n_components; ++i) exp_var->v[i][0] = D->v[i][i];
    }

    smat_free(Xc); smat_free(V); smat_free(D);
    if (projected_out) *projected_out = proj; else smat_free(proj);
    if (components_out) *components_out = W; else smat_free(W);
    if (explained_variance_out) *explained_variance_out = exp_var; else smat_free(exp_var);
    return SMAT_OK;
}

static inline smat_err_t smat_euclidean_distance(const smat_t *X, smat_t *D) {
    if (!X || !D) return SMAT_ERR_NULL;
    if (X->rows != D->rows || X->rows != D->cols) return SMAT_ERR_DIM;
    int n = X->rows;
    int p = X->cols;
    for (int i = 0; i < n; ++i) D->v[i][i] = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < p; ++k) {
                double diff = X->v[i][k] - X->v[j][k];
                s += diff * diff;
            }
            double dist = sqrt(s);
            D->v[i][j] = dist;
            D->v[j][i] = dist;
        }
    }
    return SMAT_OK;
}

static inline smat_t* smat_pdist(const smat_t *X) {
    if (!X) return NULL;
    smat_t *D = smat_create(X->rows, X->rows);
    if (!D) return NULL;
    smat_euclidean_distance(X, D);
    return D;
}

static inline smat_t* smat_mds_classical(const smat_t *D, int dim) {
    if (!smat_is_square(D) || dim < 1 || dim > D->rows) return NULL;
    int n = D->rows;
    smat_t *D2 = smat_create(n, n);
    smat_t *B = smat_create(n, n);
    double *row_means = (double*)calloc((size_t)n, sizeof(double));
    double *col_means = (double*)calloc((size_t)n, sizeof(double));
    if (!D2 || !B || !row_means || !col_means) {
        smat_free(D2); smat_free(B); free(row_means); free(col_means);
        return NULL;
    }

    double grand_mean = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double d2 = D->v[i][j] * D->v[i][j];
            D2->v[i][j] = d2;
            row_means[i] += d2;
            col_means[j] += d2;
            grand_mean += d2;
        }
    }
    for (int i = 0; i < n; ++i) {
        row_means[i] /= (double)n;
        col_means[i] /= (double)n;
    }
    grand_mean /= (double)(n * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            B->v[i][j] = -0.5 * (D2->v[i][j] - row_means[i] - col_means[j] + grand_mean);
        }
    }
    /* Symmetrize numerical rounding */
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double avg = 0.5 * (B->v[i][j] + B->v[j][i]);
            B->v[i][j] = avg;
            B->v[j][i] = avg;
        }
    }

    smat_t *V = NULL, *Lam = NULL;
    if (smat_eigen(B, &V, &Lam) != SMAT_OK) {
        smat_free(D2); smat_free(B); free(row_means); free(col_means);
        return NULL;
    }

    smat_t *X = smat_create(n, dim);
    if (X) {
        for (int j = 0; j < dim; ++j) {
            double scale = (Lam->v[j][j] > 0.0) ? sqrt(Lam->v[j][j]) : 0.0;
            for (int i = 0; i < n; ++i) {
                X->v[i][j] = V->v[i][j] * scale;
            }
        }
    }

    smat_free(D2); smat_free(B); smat_free(V); smat_free(Lam);
    free(row_means); free(col_means);
    return X;
}

static inline smat_t* smat_mds_smacof(const smat_t *D, const smat_t *X0, int dim, int iter) {
    if (!smat_is_square(D) || dim < 1 || iter < 1) return NULL;
    int n = D->rows;
    smat_t *X = NULL;
    if (X0) {
        if (X0->rows != n || X0->cols != dim) return NULL;
        X = smat_copy(X0);
    } else {
        X = smat_rand(n, dim);
        if (!X) return NULL;
        double d_mean = smat_mean(D);
        smat_add_scalar_self(X, -0.5);
        smat_mul_scalar_self(X, 0.1 * d_mean / ((1.0 / 3.0) * sqrt((double)dim)));
    }
    smat_t *Z = smat_copy(X);
    smat_t *D_ = smat_create(n, n);
    smat_t *B = smat_create(n, n);
    if (!Z || !D_ || !B) {
        smat_free(X); smat_free(Z); smat_free(D_); smat_free(B);
        return NULL;
    }

    smat_euclidean_distance(X, D_);
    for (int it = 0; it < iter; ++it) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j || fabs(D_->v[i][j]) < EPSILON) B->v[i][j] = 0.0;
                else B->v[i][j] = -D->v[i][j] / D_->v[i][j];
            }
        }
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += B->v[i][j];
            B->v[j][j] = -s;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < dim; ++j) {
                double s = 0.0;
                for (int k = 0; k < n; ++k) s += B->v[i][k] * Z->v[k][j];
                X->v[i][j] = s / (double)n;
            }
        }
        smat_euclidean_distance(X, D_);
        memcpy(Z->data, X->data, (size_t)n * (size_t)dim * sizeof(double));
    }

    smat_free(Z); smat_free(D_); smat_free(B);
    return X;
}

static inline smat_t* smat_mds_ucf(const smat_t *D, const smat_t *X0, int dim, int iter) {
    if (!smat_is_square(D) || dim < 1 || iter < 1) return NULL;
    int n = D->rows;
    smat_t *X = NULL;
    if (X0) {
        if (X0->rows != n || X0->cols != dim) return NULL;
        X = smat_copy(X0);
    } else {
        X = smat_rand(n, dim);
        if (!X) return NULL;
        double d_mean = smat_mean(D);
        smat_add_scalar_self(X, -0.5);
        smat_mul_scalar_self(X, 0.1 * d_mean / ((1.0 / 3.0) * sqrt((double)dim)));
    }
    smat_t *pmat = smat_create(n, dim);
    smat_t *dhdum = smat_create(n, 1);
    smat_t *dhmat = smat_zeros(n > 1 ? n - 1 : 1, dim);
    smat_t *RP = smat_randperm(n, iter);
    if (!pmat || !dhdum || !dhmat || !RP) {
        smat_free(X); smat_free(pmat); smat_free(dhdum); smat_free(dhmat); smat_free(RP);
        return NULL;
    }

    const double lr = 0.05;
    const double r = 2.0;

    for (int it = 0; it < iter; ++it) {
        for (int rp = 0; rp < n; ++rp) {
            int m = (int)RP->v[rp][it] - 1;
            for (int i = 0; i < n; ++i) {
                double temp = 0.0;
                for (int j = 0; j < dim; ++j) {
                    pmat->v[i][j] = X->v[m][j] - X->v[i][j];
                    temp += pow(fabs(pmat->v[i][j]), r);
                }
                dhdum->v[i][0] = pow(temp, 1.0 / r);
            }
            for (int i = 0; i < n - 1; ++i) {
                int ii = (i >= m) ? (i + 1) : i;
                double dist = dhdum->v[ii][0];
                double temp = (dist > 1e-12) ? (lr * (dist - D->v[ii][m]) * pow(dist, 1.0 - r)) : 0.0;
                for (int j = 0; j < dim; ++j) dhmat->v[i][j] = temp;
            }
            for (int i = 0; i < n - 1; ++i) {
                int ii = (i >= m) ? (i + 1) : i;
                for (int j = 0; j < dim; ++j) {
                    double p_val = pmat->v[ii][j];
                    double sgn = (p_val > 0.0) ? 1.0 : ((p_val < 0.0) ? -1.0 : 0.0);
                    X->v[ii][j] += dhmat->v[i][j] * pow(fabs(p_val), r - 1.0) * sgn;
                }
            }
        }
    }

    smat_free(pmat); smat_free(dhdum); smat_free(dhmat); smat_free(RP);
    return X;
}

#ifdef __cplusplus
} /* extern "C" */
#endif


/* ============================================================================
 * PART 2: C++ TEMPLATE API (C++11 and later)
 * ============================================================================ */

#ifdef __cplusplus

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <initializer_list>
#include <utility>
#include <random>
#include <functional>

namespace smat
{

inline std::mt19937& globalRng() {
    static std::mt19937 gen(static_cast<unsigned int>(std::random_device{}()));
    return gen;
}

inline void seed(unsigned int s) {
    globalRng().seed(s);
    smat_seed(s);
}

template <class T>
class Matrix
{
public:
  // v acts as row pointers for convenient [i][j] access
  std::vector<T*> v;

private:
  int rows_;
  int columns_;
  std::vector<T> data_; // Contiguous row-major data storage

  void syncRowPointers() {
    v.resize(rows_);
    if (rows_ > 0 && columns_ > 0 && !data_.empty()) {
      for (int i = 0; i < rows_; ++i) {
        v[i] = &data_[static_cast<size_t>(i) * static_cast<size_t>(columns_)];
      }
    } else {
      v.clear();
    }
  }

public:
  // Constructors
  Matrix();                                                        // default constructor
  Matrix(int rows, int columns);                                   // constructor with size (zero-initialized)
  Matrix(int rows, int columns, T value);                          // constructor with size and constant value
  Matrix(int rows, int columns, std::string type);                 // constructor with size and special type ("I", "rand", "randn", "rand_int", "randperm", "zeros", "ones")
  Matrix(const Matrix &other);                                     // copy constructor
  Matrix(Matrix &&other) noexcept;                                 // move constructor
  Matrix(std::initializer_list<std::initializer_list<T>> list);    // 2D initializer list
  Matrix(const char *filename);                                    // read from file
  explicit Matrix(const std::string &filename);                    // read from std::string filename
  explicit Matrix(const smat_t *c_mat);                            // construct from C smat_t

  // Static factory methods for ergonomic construction
  static Matrix<T> zeros(int rows, int columns);
  static Matrix<T> ones(int rows, int columns);
  static Matrix<T> eye(int n);
  static Matrix<T> eye(int rows, int columns);
  static Matrix<T> identity(int n);
  static Matrix<T> rand(int rows, int columns);
  static Matrix<T> randn(int rows, int columns);
  static Matrix<T> diag(const Matrix<T> &vec_or_mat);
  static Matrix<T> diag(const std::vector<T> &vec);
  static Matrix<T> linspace(T start, T end, int n);
  static Matrix<T> arange(T start, T stop, T step = static_cast<T>(1));
  static Matrix<T> fromVector(int rows, int columns, const std::vector<T> &vec);
  static Matrix<T> fromArray(int rows, int columns, const T *arr);

  // Assignment operators
  Matrix& operator=(const Matrix &other);                          // copy assignment
  Matrix& operator=(Matrix &&other) noexcept;                      // move assignment

  // Destructor
  ~Matrix();

  // Element access & properties
  void set(int r, int c, T value);
  T get(int r, int c) const;
  T& operator()(int r, int c);
  const T& operator()(int r, int c) const;
  T& operator()(int linear_idx);
  const T& operator()(int linear_idx) const;
  T* operator[](int r) { return v[r]; }
  const T* operator[](int r) const { return v[r]; }

  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }

  int rows() const;
  int columns() const;
  int cols() const { return columns_; }
  int size() const { return rows_ * columns_; }
  bool empty() const { return rows_ == 0 || columns_ == 0; }
  bool isSquare() const { return rows_ > 0 && rows_ == columns_; }
  bool isSymmetric(double tol = 1e-9) const;
  bool equals(const Matrix<T> &other, double tol = 1e-9) const;
  void fill(T value);

  // I/O & conversion
  void print() const;
  Matrix<T> copy() const;
  void saveTxt(const char *filename) const;
  void saveTxt(const std::string &filename) const { saveTxt(filename.c_str()); }
  smat_t* toCMatrix() const;

  // Shape, slicing & structural manipulation
  Matrix<T> transpose() const;
  Matrix<T> T_mat() const { return transpose(); }
  Matrix<T> reshape(int new_rows, int new_cols) const;
  Matrix<T> flatten() const;
  Matrix<T> sub(int r1, int r2, int c1, int c2) const;
  Matrix<T> row(int r) const;
  Matrix<T> col(int c) const;
  void setRow(int r, const Matrix<T> &rowVec);
  void setCol(int c, const Matrix<T> &colVec);
  void swapRows(int r1, int r2);
  void swapCols(int c1, int c2);
  Matrix<T> diag() const;
  Matrix<T> triu(int k = 0) const;
  Matrix<T> tril(int k = 0) const;

  // Concatenation
  Matrix<T> concatenateRight(const Matrix<T> &A) const;
  Matrix<T> concatenateBottom(const Matrix<T> &A) const;
  Matrix<T> hstack(const Matrix<T> &A) const { return concatenateRight(A); }
  Matrix<T> vstack(const Matrix<T> &A) const { return concatenateBottom(A); }

  // Reductions, norms & statistics
  T trace() const;
  double fnorm() const;
  double pnorm(double p) const;
  double norm1() const;
  double normInf() const;
  double norm2() const;
  double cond() const;
  T maxEl(int &r, int &c) const;
  T maxEl() const;
  T minEl(int &r, int &c) const;
  T minEl() const;
  double mean() const;
  T sum() const;
  double var(bool sample = false) const;
  double std(bool sample = false) const;
  Matrix<T> sumAxis(int axis) const;
  Matrix<double> meanAxis(int axis) const;
  Matrix<double> cov() const;
  Matrix<double> corrcoef() const;

  // In-place scalar operations
  void addNumberSelf(T value);
  void subtractNumberSelf(T value);
  void multiplyNumberSelf(T value);
  void divideNumberSelf(T value);

  // Element-wise mathematical functions
  Matrix<T> abs() const;
  Matrix<T> exp() const;
  Matrix<T> log() const;
  Matrix<T> sqrt() const;
  Matrix<T> power(double p) const;
  Matrix<T> sin() const;
  Matrix<T> cos() const;
  Matrix<T> tanh() const;
  Matrix<T> apply(std::function<T(T)> fn) const;

  // In-place element-wise matrix operations
  void addMatrixSelf(const Matrix<T> &A);
  void subtractMatrixSelf(const Matrix<T> &A);
  void dotMultiplyMatrixSelf(const Matrix<T> &A);
  void dotDivideMatrixSelf(const Matrix<T> &A);

  // New-matrix arithmetic & products
  Matrix<T> addMatrixNew(const Matrix<T> &A) const;
  Matrix<T> subtractMatrixNew(const Matrix<T> &A) const;
  Matrix<T> dotMultiplyMatrixNew(const Matrix<T> &A) const;
  Matrix<T> dotDivideMatrixNew(const Matrix<T> &A) const;
  Matrix<T> multiplyMatrixNew(const Matrix<T> &A) const;
  T dot(const Matrix<T> &other) const;
  Matrix<T> outer(const Matrix<T> &other) const;
  Matrix<T> cross(const Matrix<T> &other) const;
  Matrix<T> kron(const Matrix<T> &other) const;

  // Linear Algebra Decompositions & Solvers
  void lu(Matrix<T> &L, Matrix<T> &U, Matrix<T> &P) const;
  Matrix<T> cholesky() const;
  void cholesky(Matrix<T> &L) const;
  void qr(Matrix<T> &Q, Matrix<T> &R) const;
  void eigen(Matrix<T> &V, Matrix<T> &D) const;
  Matrix<T> eigenvalues() const;
  void svd(Matrix<T> &U, Matrix<T> &S, Matrix<T> &Vt) const;
  T determinant() const;
  T det() const { return determinant(); }
  int rank(double tol = -1.0) const;
  Matrix<T> inverse() const;
  Matrix<T> inv() const { return inverse(); }
  Matrix<T> pinv(double tol = -1.0) const;
  Matrix<T> solve(const Matrix<T> &B) const;
  Matrix<T> lstsq(const Matrix<T> &B) const;
  Matrix<T> orth(double tol = -1.0) const;
  Matrix<T> nullSpace(double tol = -1.0) const;
  Matrix<T> expm() const;
  void pca(int n_components, Matrix<T> &projected, Matrix<T> &components, Matrix<T> &explained_variance) const;

  // Operator overloading
  Matrix<T> operator+(const Matrix<T> &other) const;
  Matrix<T> operator-(const Matrix<T> &other) const;
  Matrix<T> operator*(const Matrix<T> &other) const;
  Matrix<T>& operator+=(const Matrix<T> &other);
  Matrix<T>& operator-=(const Matrix<T> &other);
  Matrix<T>& operator*=(const Matrix<T> &other);

  Matrix<T> operator-() const;
  Matrix<T> operator+(T scalar) const;
  Matrix<T> operator-(T scalar) const;
  Matrix<T> operator*(T scalar) const;
  Matrix<T> operator/(T scalar) const;
  Matrix<T>& operator+=(T scalar);
  Matrix<T>& operator-=(T scalar);
  Matrix<T>& operator*=(T scalar);
  Matrix<T>& operator/=(T scalar);

  bool operator==(const Matrix<T> &other) const;
  bool operator!=(const Matrix<T> &other) const;
};

/**********************************************
 * Utilities part
 **********************************************/

template <class T>
inline T min(T v1, T v2) { return (v1 < v2) ? v1 : v2; }

template <class T>
inline T max(T v1, T v2) { return (v1 > v2) ? v1 : v2; }

template <class T>
inline void swap(T &v1, T &v2) {
  T v3 = v1;
  v1 = v2;
  v2 = v3;
}

template <class T>
inline double sign(T v) {
  if (v > 0) return 1.0;
  else if (v < 0) return -1.0;
  else return 0.0;
}

/**********************************************
 * Implementation part
 **********************************************/

template <class T>
Matrix<T>::Matrix() : rows_(0), columns_(0) {}

template <class T>
Matrix<T>::Matrix(int rows, int columns) {
  if (rows < 1 || columns < 1) {
    throw std::invalid_argument("Invalid construction arguments: rows=" + std::to_string(rows) + ", columns=" + std::to_string(columns));
  }
  rows_ = rows;
  columns_ = columns;
  data_.assign(static_cast<size_t>(rows) * static_cast<size_t>(columns), static_cast<T>(0));
  syncRowPointers();
}

template <class T>
Matrix<T>::Matrix(int rows, int columns, T value) {
  if (rows < 1 || columns < 1) {
    throw std::invalid_argument("Invalid construction arguments: rows=" + std::to_string(rows) + ", columns=" + std::to_string(columns));
  }
  rows_ = rows;
  columns_ = columns;
  data_.assign(static_cast<size_t>(rows) * static_cast<size_t>(columns), value);
  syncRowPointers();
}

template <class T>
Matrix<T>::Matrix(int rows, int columns, std::string type) {
  if (rows < 1 || columns < 1) {
    throw std::invalid_argument("Invalid construction arguments: rows=" + std::to_string(rows) + ", columns=" + std::to_string(columns));
  }
  rows_ = rows;
  columns_ = columns;
  data_.assign(static_cast<size_t>(rows) * static_cast<size_t>(columns), static_cast<T>(0));
  syncRowPointers();

  if (type == "I" || type == "eye" || type == "identity") {
    for (int i = 0; i < std::min(rows_, columns_); i++) {
      v[i][i] = static_cast<T>(1);
    }
  } else if (type == "zeros") {
    // Already zero-initialized
  } else if (type == "ones") {
    std::fill(data_.begin(), data_.end(), static_cast<T>(1));
  } else if (type == "rand") {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    auto &rng = globalRng();
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] = static_cast<T>(dist(rng));
    }
  } else if (type == "randn") {
    std::normal_distribution<double> dist(0.0, 1.0);
    auto &rng = globalRng();
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] = static_cast<T>(dist(rng));
    }
  } else if (type == "rand_int") {
    std::uniform_int_distribution<int> dist(0, RAND_MAX);
    auto &rng = globalRng();
    for (size_t i = 0; i < data_.size(); ++i) {
      data_[i] = static_cast<T>(dist(rng));
    }
  } else if (type == "randperm") {
    auto &rng = globalRng();
    for (int j = 0; j < columns_; ++j) {
      std::vector<int> col_perm(rows_);
      for (int k = 0; k < rows_; ++k) col_perm[k] = k + 1;
      std::shuffle(col_perm.begin(), col_perm.end(), rng);
      for (int i = 0; i < rows_; ++i) v[i][j] = static_cast<T>(col_perm[i]);
    }
  } else {
    throw std::invalid_argument("Undefined matrix type: " + type);
  }
}

template <class T>
Matrix<T>::Matrix(const Matrix &other)
  : rows_(other.rows_), columns_(other.columns_), data_(other.data_) {
  syncRowPointers();
}

template <class T>
Matrix<T>::Matrix(Matrix &&other) noexcept
  : rows_(other.rows_), columns_(other.columns_), data_(std::move(other.data_)) {
  syncRowPointers();
  other.rows_ = 0;
  other.columns_ = 0;
  other.v.clear();
}

template <class T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list) {
  rows_ = static_cast<int>(list.size());
  if (rows_ == 0) {
    columns_ = 0;
    return;
  }
  columns_ = static_cast<int>(list.begin()->size());
  if (columns_ == 0) {
    rows_ = 0;
    return;
  }
  data_.resize(static_cast<size_t>(rows_) * static_cast<size_t>(columns_));
  syncRowPointers();

  int i = 0;
  for (const auto &row : list) {
    if (static_cast<int>(row.size()) != columns_) {
      throw std::invalid_argument("All rows in initializer list must have the same size.");
    }
    int j = 0;
    for (const auto &val : row) {
      v[i][j] = val;
      j++;
    }
    i++;
  }
}

template <class T>
Matrix<T>::Matrix(const char *filename) {
  if (!filename) {
    throw std::invalid_argument("Filename cannot be null.");
  }
  FILE *pFile = fopen(filename, "r");
  if (pFile == NULL) {
    throw std::runtime_error(std::string("File \"") + filename + "\" cannot be found.");
  }

  std::vector<char> line_buf(MAX_LINE_LENGTH);
  char *line = line_buf.data();
  int rows = 0;
  int columns = 0;

  while (fgets(line, MAX_LINE_LENGTH, pFile) != NULL) {
    int check = 0;
    char *token = strtok(line, " ,\t\r\n");
    while (token != NULL && token[0] >= 32) {
      check++;
      token = strtok(NULL, " ,\t\r\n");
    }
    if (check == 0) continue;
    if (rows == 0) {
      columns = check;
      rows = 1;
    } else if (check >= columns) {
      rows++;
    } else {
      break;
    }
  }
  fclose(pFile);

  if (rows < 1 || columns < 1) {
    throw std::runtime_error(std::string("File \"") + filename + "\" contains no valid matrix data.");
  }
  printf("Reading matrix from file \"%s\": %d rows, %d columns\n", filename, rows, columns);

  rows_ = rows;
  columns_ = columns;
  data_.resize(static_cast<size_t>(rows) * static_cast<size_t>(columns));
  syncRowPointers();

  pFile = fopen(filename, "r");
  if (pFile == NULL) {
    throw std::runtime_error("File cannot be found (second pass).");
  }
  int i = 0;
  while (i < rows && fgets(line, MAX_LINE_LENGTH, pFile) != NULL) {
    char *token = strtok(line, " ,\t\r\n");
    if (!token) continue;
    for (int j = 0; j < columns; j++) {
      if (j > 0) token = strtok(NULL, " ,\t\r\n");
      v[i][j] = token ? static_cast<T>(atof(token)) : static_cast<T>(0);
    }
    i++;
  }
  fclose(pFile);
}

template <class T>
Matrix<T>::Matrix(const std::string &filename) : Matrix(filename.c_str()) {}

template <class T>
Matrix<T>::Matrix(const smat_t *c_mat) {
  if (!c_mat || c_mat->rows < 1 || c_mat->cols < 1) {
    rows_ = 0;
    columns_ = 0;
    return;
  }
  rows_ = c_mat->rows;
  columns_ = c_mat->cols;
  size_t total = static_cast<size_t>(rows_) * static_cast<size_t>(columns_);
  data_.resize(total);
  for (size_t i = 0; i < total; ++i) {
    data_[i] = static_cast<T>(c_mat->data[i]);
  }
  syncRowPointers();
}

template <class T>
Matrix<T> Matrix<T>::zeros(int rows, int columns) {
  return Matrix<T>(rows, columns, static_cast<T>(0));
}

template <class T>
Matrix<T> Matrix<T>::ones(int rows, int columns) {
  return Matrix<T>(rows, columns, static_cast<T>(1));
}

template <class T>
Matrix<T> Matrix<T>::eye(int n) {
  return Matrix<T>(n, n, "I");
}

template <class T>
Matrix<T> Matrix<T>::eye(int rows, int columns) {
  return Matrix<T>(rows, columns, "I");
}

template <class T>
Matrix<T> Matrix<T>::identity(int n) {
  return Matrix<T>(n, n, "I");
}

template <class T>
Matrix<T> Matrix<T>::rand(int rows, int columns) {
  return Matrix<T>(rows, columns, "rand");
}

template <class T>
Matrix<T> Matrix<T>::randn(int rows, int columns) {
  return Matrix<T>(rows, columns, "randn");
}

template <class T>
Matrix<T> Matrix<T>::diag(const Matrix<T> &vec_or_mat) {
  return vec_or_mat.diag();
}

template <class T>
Matrix<T> Matrix<T>::diag(const std::vector<T> &vec) {
  int n = static_cast<int>(vec.size());
  if (n < 1) throw std::invalid_argument("Vector for diag() cannot be empty.");
  Matrix<T> D(n, n, static_cast<T>(0));
  for (int i = 0; i < n; ++i) D.v[i][i] = vec[i];
  return D;
}

template <class T>
Matrix<T> Matrix<T>::linspace(T start, T end, int n) {
  if (n < 1) throw std::invalid_argument("linspace requires n >= 1.");
  Matrix<T> B(1, n);
  if (n == 1) {
    B.v[0][0] = end;
    return B;
  }
  double step = (static_cast<double>(end) - static_cast<double>(start)) / static_cast<double>(n - 1);
  for (int i = 0; i < n; ++i) {
    B.v[0][i] = static_cast<T>(static_cast<double>(start) + i * step);
  }
  B.v[0][n - 1] = end;
  return B;
}

template <class T>
Matrix<T> Matrix<T>::arange(T start, T stop, T step) {
  if (static_cast<double>(step) == 0.0) throw std::invalid_argument("arange step cannot be zero.");
  double d_start = static_cast<double>(start);
  double d_stop = static_cast<double>(stop);
  double d_step = static_cast<double>(step);
  if ((d_stop > d_start && d_step < 0.0) || (d_stop < d_start && d_step > 0.0)) {
    throw std::invalid_argument("arange step direction does not match start/stop.");
  }
  int n = static_cast<int>(std::ceil((d_stop - d_start) / d_step - 1e-12));
  if (n < 1) throw std::invalid_argument("arange resulted in empty range.");
  Matrix<T> B(1, n);
  for (int i = 0; i < n; ++i) {
    B.v[0][i] = static_cast<T>(d_start + i * d_step);
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::fromVector(int rows, int columns, const std::vector<T> &vec) {
  if (rows < 1 || columns < 1 || static_cast<int>(vec.size()) != rows * columns) {
    throw std::invalid_argument("Vector size does not match matrix dimensions.");
  }
  Matrix<T> B(rows, columns);
  B.data_ = vec;
  B.syncRowPointers();
  return B;
}

template <class T>
Matrix<T> Matrix<T>::fromArray(int rows, int columns, const T *arr) {
  if (!arr || rows < 1 || columns < 1) {
    throw std::invalid_argument("Invalid arguments to fromArray.");
  }
  Matrix<T> B(rows, columns);
  std::copy(arr, arr + rows * columns, B.data_.begin());
  return B;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {
  if (this != &other) {
    rows_ = other.rows_;
    columns_ = other.columns_;
    data_ = other.data_;
    syncRowPointers();
  }
  return *this;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other) noexcept {
  if (this != &other) {
    rows_ = other.rows_;
    columns_ = other.columns_;
    data_ = std::move(other.data_);
    syncRowPointers();
    other.rows_ = 0;
    other.columns_ = 0;
    other.v.clear();
  }
  return *this;
}

template <class T>
Matrix<T>::~Matrix() {}

template <class T>
void Matrix<T>::set(int r, int c, T value) {
  if (r < 0 || r >= rows_ || c < 0 || c >= columns_) {
    throw std::out_of_range("Invalid index in set(): r=" + std::to_string(r) + ", c=" + std::to_string(c));
  }
  v[r][c] = value;
}

template <class T>
T Matrix<T>::get(int r, int c) const {
  if (r < 0 || r >= rows_ || c < 0 || c >= columns_) {
    throw std::out_of_range("Matrix index out of range in get().");
  }
  return v[r][c];
}

template <class T>
T& Matrix<T>::operator()(int r, int c) {
  if (r < 0 || r >= rows_ || c < 0 || c >= columns_) {
    throw std::out_of_range("Matrix index out of range in operator()(r, c).");
  }
  return v[r][c];
}

template <class T>
const T& Matrix<T>::operator()(int r, int c) const {
  if (r < 0 || r >= rows_ || c < 0 || c >= columns_) {
    throw std::out_of_range("Matrix index out of range in operator()(r, c).");
  }
  return v[r][c];
}

template <class T>
T& Matrix<T>::operator()(int linear_idx) {
  if (linear_idx < 0 || linear_idx >= rows_ * columns_) {
    throw std::out_of_range("Matrix index out of range in operator()(idx).");
  }
  return data_[linear_idx];
}

template <class T>
const T& Matrix<T>::operator()(int linear_idx) const {
  if (linear_idx < 0 || linear_idx >= rows_ * columns_) {
    throw std::out_of_range("Matrix index out of range in operator()(idx).");
  }
  return data_[linear_idx];
}

template <class T>
int Matrix<T>::rows() const { return rows_; }

template <class T>
int Matrix<T>::columns() const { return columns_; }

template <class T>
bool Matrix<T>::isSymmetric(double tol) const {
  if (!isSquare()) return false;
  for (int i = 0; i < rows_; ++i) {
    for (int j = i + 1; j < columns_; ++j) {
      if (std::abs(static_cast<double>(v[i][j]) - static_cast<double>(v[j][i])) > tol) {
        return false;
      }
    }
  }
  return true;
}

template <class T>
bool Matrix<T>::equals(const Matrix<T> &other, double tol) const {
  if (rows_ != other.rows_ || columns_ != other.columns_) return false;
  for (size_t i = 0; i < data_.size(); ++i) {
    if (std::abs(static_cast<double>(data_[i]) - static_cast<double>(other.data_[i])) > tol) {
      return false;
    }
  }
  return true;
}

template <class T>
void Matrix<T>::fill(T value) {
  std::fill(data_.begin(), data_.end(), value);
}

template <class T>
void Matrix<T>::print() const {
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) {
      if (std::abs(static_cast<double>(v[i][j])) < EPSILON)
        printf("%10.4f ", 0.0);
      else
        printf("%10.4f ", static_cast<double>(v[i][j]));
    }
    printf("\n");
  }
}

template <class T>
Matrix<T> Matrix<T>::copy() const {
  return Matrix<T>(*this);
}

template <class T>
void Matrix<T>::saveTxt(const char *filename) const {
  if (!filename) throw std::invalid_argument("Filename cannot be null.");
  FILE *pFile = fopen(filename, "w");
  if (pFile == NULL) {
    throw std::runtime_error("File cannot be opened for writing.");
  }
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) {
      fprintf(pFile, "%f ", static_cast<double>(v[i][j]));
    }
    fprintf(pFile, "\n");
  }
  fclose(pFile);
  printf("Matrix saved to file \"%s\"\n", filename);
}

template <class T>
smat_t* Matrix<T>::toCMatrix() const {
  if (empty()) return NULL;
  smat_t *cm = smat_create(rows_, columns_);
  if (!cm) return NULL;
  for (size_t i = 0; i < data_.size(); ++i) {
    cm->data[i] = static_cast<double>(data_[i]);
  }
  return cm;
}

template <class T>
Matrix<T> Matrix<T>::transpose() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(columns_, rows_);
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) {
      B.v[j][i] = v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::reshape(int new_rows, int new_cols) const {
  if (new_rows < 1 || new_cols < 1 || new_rows * new_cols != rows_ * columns_) {
    throw std::invalid_argument("Invalid reshape dimensions.");
  }
  return Matrix<T>::fromVector(new_rows, new_cols, data_);
}

template <class T>
Matrix<T> Matrix<T>::flatten() const {
  if (empty()) return Matrix<T>();
  return reshape(rows_ * columns_, 1);
}

template <class T>
Matrix<T> Matrix<T>::sub(int r1, int r2, int c1, int c2) const {
  if (r1 < 0 || r2 >= rows_ || c1 < 0 || c2 >= columns_ || r1 > r2 || c1 > c2) {
    throw std::out_of_range("Matrix index out of range in sub().");
  }
  Matrix<T> B(r2 - r1 + 1, c2 - c1 + 1);
  for (int i = r1; i <= r2; i++) {
    for (int j = c1; j <= c2; j++) {
      B.v[i - r1][j - c1] = v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::row(int r) const {
  return sub(r, r, 0, columns_ - 1);
}

template <class T>
Matrix<T> Matrix<T>::col(int c) const {
  return sub(0, rows_ - 1, c, c);
}

template <class T>
void Matrix<T>::setRow(int r, const Matrix<T> &rowVec) {
  if (r < 0 || r >= rows_) throw std::out_of_range("Row index out of range in setRow().");
  if (rowVec.size() != columns_) throw std::invalid_argument("Dimension mismatch in setRow().");
  for (int j = 0; j < columns_; ++j) v[r][j] = rowVec.data_[j];
}

template <class T>
void Matrix<T>::setCol(int c, const Matrix<T> &colVec) {
  if (c < 0 || c >= columns_) throw std::out_of_range("Column index out of range in setCol().");
  if (colVec.size() != rows_) throw std::invalid_argument("Dimension mismatch in setCol().");
  for (int i = 0; i < rows_; ++i) v[i][c] = colVec.data_[i];
}

template <class T>
void Matrix<T>::swapRows(int r1, int r2) {
  if (r1 < 0 || r1 >= rows_ || r2 < 0 || r2 >= rows_) {
    throw std::out_of_range("Row index out of range in swapRows().");
  }
  if (r1 == r2) return;
  for (int j = 0; j < columns_; ++j) {
    std::swap(v[r1][j], v[r2][j]);
  }
}

template <class T>
void Matrix<T>::swapCols(int c1, int c2) {
  if (c1 < 0 || c1 >= columns_ || c2 < 0 || c2 >= columns_) {
    throw std::out_of_range("Column index out of range in swapCols().");
  }
  if (c1 == c2) return;
  for (int i = 0; i < rows_; ++i) {
    std::swap(v[i][c1], v[i][c2]);
  }
}

template <class T>
Matrix<T> Matrix<T>::diag() const {
  if (empty()) throw std::runtime_error("Cannot call diag() on empty matrix.");
  if ((rows_ == 1 && columns_ > 1) || (columns_ == 1 && rows_ > 1)) {
    int n = std::max(rows_, columns_);
    Matrix<T> D(n, n, static_cast<T>(0));
    for (int i = 0; i < n; ++i) D.v[i][i] = data_[i];
    return D;
  } else {
    int k = std::min(rows_, columns_);
    Matrix<T> d(k, 1);
    for (int i = 0; i < k; ++i) d.v[i][0] = v[i][i];
    return d;
  }
}

template <class T>
Matrix<T> Matrix<T>::triu(int k) const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_, static_cast<T>(0));
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < columns_; ++j) {
      if (j - i >= k) B.v[i][j] = v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::tril(int k) const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_, static_cast<T>(0));
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < columns_; ++j) {
      if (j - i <= k) B.v[i][j] = v[i][j];
    }
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::abs() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::abs(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
T Matrix<T>::trace() const {
  T s = 0;
  for (int i = 0; i < std::min(rows_, columns_); i++)
    s += v[i][i];
  return s;
}

template <class T>
double Matrix<T>::fnorm() const {
  double s = 0;
  for (size_t i = 0; i < data_.size(); ++i) {
    double val = static_cast<double>(data_[i]);
    s += val * val;
  }
  return std::sqrt(s);
}

template <class T>
double Matrix<T>::pnorm(double p) const {
  if (p <= 0.0) throw std::invalid_argument("pnorm requires p > 0.");
  double s = 0;
  for (size_t i = 0; i < data_.size(); ++i) {
    s += std::pow(std::fabs(static_cast<double>(data_[i])), p);
  }
  return std::pow(s, 1.0 / p);
}

template <class T>
double Matrix<T>::norm1() const {
  double max_col = 0.0;
  for (int j = 0; j < columns_; ++j) {
    double col_sum = 0.0;
    for (int i = 0; i < rows_; ++i) col_sum += std::abs(static_cast<double>(v[i][j]));
    if (col_sum > max_col) max_col = col_sum;
  }
  return max_col;
}

template <class T>
double Matrix<T>::normInf() const {
  double max_row = 0.0;
  for (int i = 0; i < rows_; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < columns_; ++j) row_sum += std::abs(static_cast<double>(v[i][j]));
    if (row_sum > max_row) max_row = row_sum;
  }
  return max_row;
}

template <class T>
double Matrix<T>::norm2() const {
  if (empty()) return 0.0;
  Matrix<T> U, S, Vt;
  this->svd(U, S, Vt);
  return static_cast<double>(S.v[0][0]);
}

template <class T>
double Matrix<T>::cond() const {
  if (empty()) throw std::runtime_error("Cannot compute cond() of empty matrix.");
  Matrix<T> U, S, Vt;
  this->svd(U, S, Vt);
  int k = std::min(rows_, columns_);
  double s_max = static_cast<double>(S.v[0][0]);
  double s_min = static_cast<double>(S.v[k - 1][k - 1]);
  if (s_min <= 1e-15 * s_max) return INFINITY;
  return s_max / s_min;
}

template <class T>
T Matrix<T>::maxEl(int &r, int &c) const {
  if (empty()) throw std::runtime_error("Cannot call maxEl() on empty matrix.");
  T max_val = v[0][0];
  r = 0;
  c = 0;
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) {
      if (v[i][j] > max_val) {
        max_val = v[i][j];
        r = i;
        c = j;
      }
    }
  }
  return max_val;
}

template <class T>
T Matrix<T>::maxEl() const {
  int r, c;
  return maxEl(r, c);
}

template <class T>
T Matrix<T>::minEl(int &r, int &c) const {
  if (empty()) throw std::runtime_error("Cannot call minEl() on empty matrix.");
  T min_val = v[0][0];
  r = 0;
  c = 0;
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) {
      if (v[i][j] < min_val) {
        min_val = v[i][j];
        r = i;
        c = j;
      }
    }
  }
  return min_val;
}

template <class T>
T Matrix<T>::minEl() const {
  int r, c;
  return minEl(r, c);
}

template <class T>
double Matrix<T>::mean() const {
  if (empty()) throw std::runtime_error("Cannot call mean() on empty matrix.");
  return static_cast<double>(this->sum()) / static_cast<double>(rows_ * columns_);
}

template <class T>
T Matrix<T>::sum() const {
  T s = 0;
  for (size_t i = 0; i < data_.size(); ++i) {
    s += data_[i];
  }
  return s;
}

template <class T>
double Matrix<T>::var(bool sample) const {
  if (empty()) throw std::runtime_error("Cannot call var() on empty matrix.");
  int n = rows_ * columns_;
  if (sample && n <= 1) throw std::invalid_argument("Sample variance requires at least 2 elements.");
  double m = mean();
  double s = 0;
  for (size_t i = 0; i < data_.size(); ++i) {
    double diff = static_cast<double>(data_[i]) - m;
    s += diff * diff;
  }
  return s / static_cast<double>(sample ? (n - 1) : n);
}

template <class T>
double Matrix<T>::std(bool sample) const {
  return std::sqrt(var(sample));
}

template <class T>
Matrix<T> Matrix<T>::sumAxis(int axis) const {
  if (empty()) throw std::runtime_error("Cannot call sumAxis() on empty matrix.");
  if (axis == 0) {
    Matrix<T> S(1, columns_, static_cast<T>(0));
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < columns_; ++j) S.v[0][j] += v[i][j];
    }
    return S;
  } else if (axis == 1) {
    Matrix<T> S(rows_, 1, static_cast<T>(0));
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < columns_; ++j) S.v[i][0] += v[i][j];
    }
    return S;
  } else {
    throw std::invalid_argument("axis must be 0 (column-wise) or 1 (row-wise).");
  }
}

template <class T>
Matrix<double> Matrix<T>::meanAxis(int axis) const {
  if (empty()) throw std::runtime_error("Cannot call meanAxis() on empty matrix.");
  if (axis == 0) {
    Matrix<double> M(1, columns_, 0.0);
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < columns_; ++j) M.v[0][j] += static_cast<double>(v[i][j]);
    }
    M.divideNumberSelf(static_cast<double>(rows_));
    return M;
  } else if (axis == 1) {
    Matrix<double> M(rows_, 1, 0.0);
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < columns_; ++j) M.v[i][0] += static_cast<double>(v[i][j]);
    }
    M.divideNumberSelf(static_cast<double>(columns_));
    return M;
  } else {
    throw std::invalid_argument("axis must be 0 (column-wise) or 1 (row-wise).");
  }
}

template <class T>
Matrix<double> Matrix<T>::cov() const {
  if (empty()) throw std::runtime_error("Cannot call cov() on empty matrix.");
  smat_t *cm = toCMatrix();
  smat_t *ccov = smat_cov(cm);
  smat_free(cm);
  Matrix<double> res(ccov);
  smat_free(ccov);
  return res;
}

template <class T>
Matrix<double> Matrix<T>::corrcoef() const {
  if (empty()) throw std::runtime_error("Cannot call corrcoef() on empty matrix.");
  smat_t *cm = toCMatrix();
  smat_t *ccorr = smat_corrcoef(cm);
  smat_free(cm);
  Matrix<double> res(ccorr);
  smat_free(ccorr);
  return res;
}

template <class T>
void Matrix<T>::addNumberSelf(T value) {
  for (size_t i = 0; i < data_.size(); ++i) data_[i] += value;
}

template <class T>
void Matrix<T>::subtractNumberSelf(T value) {
  for (size_t i = 0; i < data_.size(); ++i) data_[i] -= value;
}

template <class T>
void Matrix<T>::multiplyNumberSelf(T value) {
  for (size_t i = 0; i < data_.size(); ++i) data_[i] *= value;
}

template <class T>
void Matrix<T>::divideNumberSelf(T value) {
  if (static_cast<double>(value) == 0.0) {
    throw std::invalid_argument("Division by zero in divideNumberSelf().");
  }
  for (size_t i = 0; i < data_.size(); ++i) data_[i] /= value;
}

template <class T>
Matrix<T> Matrix<T>::exp() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::exp(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::log() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::log(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::sqrt() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::sqrt(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::power(double p) const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::pow(static_cast<double>(data_[i]), p));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::sin() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::sin(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::cos() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::cos(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::tanh() const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = static_cast<T>(std::tanh(static_cast<double>(data_[i])));
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::apply(std::function<T(T)> fn) const {
  if (empty()) return Matrix<T>();
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) {
    B.data_[i] = fn(data_[i]);
  }
  return B;
}

template <class T>
void Matrix<T>::addMatrixSelf(const Matrix<T> &A) {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Unmatched matrix sizes in matrix summation.");
  }
  for (size_t i = 0; i < data_.size(); ++i) data_[i] += A.data_[i];
}

template <class T>
void Matrix<T>::subtractMatrixSelf(const Matrix<T> &A) {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Unmatched matrix sizes in matrix subtraction.");
  }
  for (size_t i = 0; i < data_.size(); ++i) data_[i] -= A.data_[i];
}

template <class T>
void Matrix<T>::dotMultiplyMatrixSelf(const Matrix<T> &A) {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Unmatched matrix sizes in matrix dot multiplication.");
  }
  for (size_t i = 0; i < data_.size(); ++i) data_[i] *= A.data_[i];
}

template <class T>
void Matrix<T>::dotDivideMatrixSelf(const Matrix<T> &A) {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Unmatched matrix sizes in matrix dot division.");
  }
  for (size_t i = 0; i < data_.size(); ++i) data_[i] /= A.data_[i];
}

template <class T>
Matrix<T> Matrix<T>::addMatrixNew(const Matrix<T> &A) const {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Matrix dimensions must match for addition.");
  }
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) B.data_[i] = data_[i] + A.data_[i];
  return B;
}

template <class T>
Matrix<T> Matrix<T>::subtractMatrixNew(const Matrix<T> &A) const {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Matrix dimensions must match for subtraction.");
  }
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) B.data_[i] = data_[i] - A.data_[i];
  return B;
}

template <class T>
Matrix<T> Matrix<T>::dotMultiplyMatrixNew(const Matrix<T> &A) const {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Matrix dimensions must match for dot multiplication.");
  }
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) B.data_[i] = data_[i] * A.data_[i];
  return B;
}

template <class T>
Matrix<T> Matrix<T>::dotDivideMatrixNew(const Matrix<T> &A) const {
  if (rows_ != A.rows() || columns_ != A.columns()) {
    throw std::invalid_argument("Matrix dimensions must match for dot division.");
  }
  Matrix<T> B(rows_, columns_);
  for (size_t i = 0; i < data_.size(); ++i) B.data_[i] = data_[i] / A.data_[i];
  return B;
}

/* Cache-friendly i-k-j matrix multiplication */
template <class T>
Matrix<T> Matrix<T>::multiplyMatrixNew(const Matrix<T> &A) const {
  if (columns_ != A.rows()) {
    throw std::invalid_argument("Matrix dimensions must match for multiplication.");
  }
  int a_cols = A.columns();
  Matrix<T> B(rows_, a_cols, static_cast<T>(0));
  for (int i = 0; i < rows_; i++) {
    T *b_row = B.v[i];
    const T *this_row = v[i];
    for (int k = 0; k < columns_; k++) {
      T aik = this_row[k];
      const T *a_row = A.v[k];
      for (int j = 0; j < a_cols; j++) {
        b_row[j] += aik * a_row[j];
      }
    }
  }
  return B;
}

template <class T>
T Matrix<T>::dot(const Matrix<T> &other) const {
  if (size() != other.size()) {
    throw std::invalid_argument("Sizes must match for dot().");
  }
  T s = 0;
  for (size_t i = 0; i < data_.size(); ++i) s += data_[i] * other.data_[i];
  return s;
}

template <class T>
Matrix<T> Matrix<T>::outer(const Matrix<T> &other) const {
  if (empty() || other.empty()) throw std::invalid_argument("Cannot compute outer() of empty matrix.");
  int m = size();
  int n = other.size();
  Matrix<T> C(m, n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C.v[i][j] = data_[i] * other.data_[j];
    }
  }
  return C;
}

template <class T>
Matrix<T> Matrix<T>::cross(const Matrix<T> &other) const {
  if (size() != 3 || other.size() != 3) {
    throw std::invalid_argument("cross() requires 3-element vectors.");
  }
  Matrix<T> c(rows_, columns_);
  c.data_[0] = data_[1] * other.data_[2] - data_[2] * other.data_[1];
  c.data_[1] = data_[2] * other.data_[0] - data_[0] * other.data_[2];
  c.data_[2] = data_[0] * other.data_[1] - data_[1] * other.data_[0];
  return c;
}

template <class T>
Matrix<T> Matrix<T>::kron(const Matrix<T> &other) const {
  if (empty() || other.empty()) throw std::invalid_argument("Cannot compute kron() of empty matrix.");
  Matrix<T> K(rows_ * other.rows_, columns_ * other.columns_);
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < columns_; ++j) {
      T a_ij = v[i][j];
      for (int r = 0; r < other.rows_; ++r) {
        for (int c = 0; c < other.columns_; ++c) {
          K.v[i * other.rows_ + r][j * other.columns_ + c] = a_ij * other.v[r][c];
        }
      }
    }
  }
  return K;
}

template <class T>
Matrix<T> Matrix<T>::concatenateRight(const Matrix<T> &A) const {
  if (rows_ != A.rows()) {
    throw std::invalid_argument("Matrix rows must match for concatenateRight().");
  }
  Matrix<T> B(rows_, columns_ + A.columns());
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < columns_; j++) B.v[i][j] = v[i][j];
    for (int j = 0; j < A.columns(); j++) B.v[i][j + columns_] = A.v[i][j];
  }
  return B;
}

template <class T>
Matrix<T> Matrix<T>::concatenateBottom(const Matrix<T> &A) const {
  if (columns_ != A.columns()) {
    throw std::invalid_argument("Matrix columns must match for concatenateBottom().");
  }
  Matrix<T> B(rows_ + A.rows(), columns_);
  for (int j = 0; j < columns_; j++) {
    for (int i = 0; i < rows_; i++) B.v[i][j] = v[i][j];
    for (int i = 0; i < A.rows(); i++) B.v[i + rows_][j] = A.v[i][j];
  }
  return B;
}

template <class T>
void Matrix<T>::lu(Matrix<T> &L, Matrix<T> &U, Matrix<T> &P) const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("LU decomposition requires square matrix.");
  }
  int n = rows_;
  L = Matrix<T>(n, n, static_cast<T>(0));
  U = Matrix<T>(n, n, static_cast<T>(0));
  P = Matrix<T>(n, n, "I");

  Matrix<double> A(n, n);
  for (size_t idx = 0; idx < data_.size(); ++idx) A.data()[idx] = static_cast<double>(data_[idx]);

  for (int i = 0; i < n; i++) {
    double maxElVal = std::abs(A.v[i][i]);
    int maxRow = i;
    for (int k = i + 1; k < n; k++) {
      if (std::abs(A.v[k][i]) > maxElVal) {
        maxElVal = std::abs(A.v[k][i]);
        maxRow = k;
      }
    }

    if (maxRow != i) {
      A.swapRows(i, maxRow);
      P.swapRows(i, maxRow);
      for (int k = 0; k < i; k++) {
        std::swap(L.v[i][k], L.v[maxRow][k]);
      }
    }

    L.v[i][i] = static_cast<T>(1);

    if (std::abs(A.v[i][i]) > 1e-15) {
      for (int k = i + 1; k < n; k++) {
        double c = -A.v[k][i] / A.v[i][i];
        L.v[k][i] = static_cast<T>(-c);
        A.v[k][i] = 0.0;
        for (int j = i + 1; j < n; j++) {
          A.v[k][j] += c * A.v[i][j];
        }
      }
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      U.v[i][j] = static_cast<T>(A.v[i][j]);
    }
  }
}

template <class T>
Matrix<T> Matrix<T>::cholesky() const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("Cholesky decomposition requires a square matrix.");
  }
  if (!isSymmetric(1e-7)) {
    throw std::invalid_argument("Cholesky decomposition requires a symmetric matrix.");
  }
  int n = rows_;
  Matrix<T> L(n, n, static_cast<T>(0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = static_cast<double>(v[i][j]);
      for (int k = 0; k < j; ++k) {
        sum -= static_cast<double>(L.v[i][k]) * static_cast<double>(L.v[j][k]);
      }
      if (i == j) {
        if (sum <= 1e-15) {
          throw std::runtime_error("Matrix is not positive definite in cholesky().");
        }
        L.v[i][i] = static_cast<T>(std::sqrt(sum));
      } else {
        L.v[i][j] = static_cast<T>(sum / static_cast<double>(L.v[j][j]));
      }
    }
  }
  return L;
}

template <class T>
void Matrix<T>::cholesky(Matrix<T> &L) const {
  L = this->cholesky();
}

template <class T>
T Matrix<T>::determinant() const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("Determinant requires square matrix.");
  }
  smat_t *cm = toCMatrix();
  double d = smat_determinant(cm);
  smat_free(cm);
  return static_cast<T>(d);
}

template <class T>
Matrix<T> Matrix<T>::inverse() const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("Inverse requires square matrix.");
  }
  smat_t *cm = toCMatrix();
  smat_t *cinv = smat_inverse(cm);
  smat_free(cm);
  if (!cinv) {
    throw std::runtime_error("Matrix is singular, cannot find inverse.");
  }
  Matrix<T> Inv(cinv);
  smat_free(cinv);
  return Inv;
}

template <class T>
Matrix<T> Matrix<T>::pinv(double tol) const {
  if (empty()) throw std::invalid_argument("Cannot compute pinv() of empty matrix.");
  Matrix<T> U, S, Vt;
  this->svd(U, S, Vt);
  int k = S.rows();
  double sigma_max = (k > 0) ? static_cast<double>(S.v[0][0]) : 0.0;
  double cutoff = (tol >= 0.0) ? tol : (static_cast<double>(std::max(rows_, columns_)) * std::max(sigma_max, 1.0) * 1e-12);
  Matrix<T> V = Vt.transpose();
  Matrix<T> Ut = U.transpose();
  for (int i = 0; i < k; ++i) {
    double s = static_cast<double>(S.v[i][i]);
    double s_inv = (s > cutoff) ? (1.0 / s) : 0.0;
    for (int j = 0; j < V.rows(); ++j) {
      V.v[j][i] = static_cast<T>(static_cast<double>(V.v[j][i]) * s_inv);
    }
  }
  return V * Ut;
}

template <class T>
Matrix<T> Matrix<T>::solve(const Matrix<T> &B) const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("Matrix must be square to solve system.");
  }
  if (rows_ != B.rows()) {
    throw std::invalid_argument("Matrix row dimensions must agree.");
  }
  smat_t *cA = toCMatrix();
  smat_t *cB = B.toCMatrix();
  smat_t *cX = smat_solve(cA, cB);
  smat_free(cA);
  smat_free(cB);
  if (!cX) {
    throw std::runtime_error("Matrix is singular.");
  }
  Matrix<T> X(cX);
  smat_free(cX);
  return X;
}

template <class T>
Matrix<T> Matrix<T>::lstsq(const Matrix<T> &B) const {
  if (empty() || B.empty() || rows_ != B.rows()) {
    throw std::invalid_argument("Matrix row dimensions must agree in lstsq().");
  }
  return this->pinv() * B;
}

template <class T>
void Matrix<T>::qr(Matrix<T> &Q, Matrix<T> &R) const {
  if (empty()) {
    throw std::invalid_argument("Cannot compute QR of empty matrix.");
  }
  smat_t *cA = toCMatrix();
  smat_t *cQ = NULL, *cR = NULL;
  smat_err_t err = smat_qr(cA, &cQ, &cR);
  smat_free(cA);
  if (err != SMAT_OK) {
    throw std::runtime_error(smat_strerror(err));
  }
  Q = Matrix<T>(cQ);
  R = Matrix<T>(cR);
  smat_free(cQ);
  smat_free(cR);
}

template <class T>
void Matrix<T>::eigen(Matrix<T> &V, Matrix<T> &D) const {
  if (rows_ != columns_ || rows_ < 1) {
    throw std::invalid_argument("Eigen decomposition requires square matrix.");
  }
  if (!isSymmetric(1e-7)) {
    throw std::invalid_argument("Eigen decomposition requires a symmetric matrix.");
  }
  smat_t *cA = toCMatrix();
  smat_t *cV = NULL, *cD = NULL;
  smat_err_t err = smat_eigen(cA, &cV, &cD);
  smat_free(cA);
  if (err != SMAT_OK) {
    throw std::runtime_error(smat_strerror(err));
  }
  V = Matrix<T>(cV);
  D = Matrix<T>(cD);
  smat_free(cV);
  smat_free(cD);
}

template <class T>
Matrix<T> Matrix<T>::eigenvalues() const {
  Matrix<T> V, D;
  this->eigen(V, D);
  Matrix<T> eigVals(rows_, 1);
  for (int i = 0; i < rows_; i++) {
    eigVals.v[i][0] = D.v[i][i];
  }
  return eigVals;
}

template <class T>
void Matrix<T>::svd(Matrix<T> &U, Matrix<T> &S, Matrix<T> &Vt) const {
  if (empty()) {
    throw std::invalid_argument("Cannot compute SVD of empty matrix.");
  }
  smat_t *cA = toCMatrix();
  smat_t *cU = NULL, *cS = NULL, *cVt = NULL;
  smat_err_t err = smat_svd(cA, &cU, &cS, &cVt);
  smat_free(cA);
  if (err != SMAT_OK) {
    throw std::runtime_error(smat_strerror(err));
  }
  U = Matrix<T>(cU);
  S = Matrix<T>(cS);
  Vt = Matrix<T>(cVt);
  smat_free(cU);
  smat_free(cS);
  smat_free(cVt);
}

template <class T>
int Matrix<T>::rank(double tol) const {
  if (empty()) return 0;
  Matrix<T> U, S, Vt;
  this->svd(U, S, Vt);
  int k = std::min(rows_, columns_);
  double sigma_max = (k > 0) ? static_cast<double>(S.v[0][0]) : 0.0;
  double cutoff = (tol >= 0.0) ? tol : (static_cast<double>(std::max(rows_, columns_)) * std::max(sigma_max, 1.0) * 1e-12);
  int r = 0;
  for (int i = 0; i < k; i++) {
    if (std::abs(static_cast<double>(S.v[i][i])) > cutoff) r++;
  }
  return r;
}

template <class T>
Matrix<T> Matrix<T>::orth(double tol) const {
  if (empty()) return Matrix<T>();
  Matrix<T> U, S, Vt;
  this->svd(U, S, Vt);
  int r = this->rank(tol);
  if (r == 0) return Matrix<T>();
  return U.sub(0, rows_ - 1, 0, r - 1);
}

template <class T>
Matrix<T> Matrix<T>::nullSpace(double tol) const {
  if (empty()) return Matrix<T>();
  smat_t *cA = toCMatrix();
  smat_t *cN = smat_null_space(cA);
  smat_free(cA);
  (void)tol;
  if (!cN) return Matrix<T>();
  Matrix<T> N(cN);
  smat_free(cN);
  return N;
}

template <class T>
Matrix<T> Matrix<T>::expm() const {
  if (!isSquare()) {
    throw std::invalid_argument("Matrix exponential requires a square matrix.");
  }
  smat_t *cA = toCMatrix();
  smat_t *cE = smat_expm(cA);
  smat_free(cA);
  if (!cE) {
    throw std::runtime_error("Failed to compute matrix exponential.");
  }
  Matrix<T> E(cE);
  smat_free(cE);
  return E;
}

template <class T>
void Matrix<T>::pca(int n_components, Matrix<T> &projected, Matrix<T> &components, Matrix<T> &explained_variance) const {
  if (empty() || n_components < 1 || n_components > columns_) {
    throw std::invalid_argument("Invalid n_components for pca().");
  }
  smat_t *cX = toCMatrix();
  smat_t *cProj = NULL, *cComp = NULL, *cVar = NULL;
  smat_err_t err = smat_pca(cX, n_components, &cProj, &cComp, &cVar);
  smat_free(cX);
  if (err != SMAT_OK) {
    throw std::runtime_error(smat_strerror(err));
  }
  projected = Matrix<T>(cProj);
  components = Matrix<T>(cComp);
  explained_variance = Matrix<T>(cVar);
  smat_free(cProj);
  smat_free(cComp);
  smat_free(cVar);
}

/* Operator overloads */

template <class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const {
  return this->addMatrixNew(other);
}

template <class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) const {
  return this->subtractMatrixNew(other);
}

template <class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const {
  return this->multiplyMatrixNew(other);
}

template <class T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T> &other) {
  this->addMatrixSelf(other);
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T> &other) {
  this->subtractMatrixSelf(other);
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T> &other) {
  *this = this->multiplyMatrixNew(other);
  return *this;
}

template <class T>
Matrix<T> Matrix<T>::operator-() const {
  Matrix<T> B = this->copy();
  B.multiplyNumberSelf(static_cast<T>(-1));
  return B;
}

template <class T>
Matrix<T> Matrix<T>::operator+(T scalar) const {
  Matrix<T> B = this->copy();
  B.addNumberSelf(scalar);
  return B;
}

template <class T>
Matrix<T> Matrix<T>::operator-(T scalar) const {
  Matrix<T> B = this->copy();
  B.subtractNumberSelf(scalar);
  return B;
}

template <class T>
Matrix<T> Matrix<T>::operator*(T scalar) const {
  Matrix<T> B = this->copy();
  B.multiplyNumberSelf(scalar);
  return B;
}

template <class T>
Matrix<T> Matrix<T>::operator/(T scalar) const {
  Matrix<T> B = this->copy();
  B.divideNumberSelf(scalar);
  return B;
}

template <class T>
Matrix<T>& Matrix<T>::operator+=(T scalar) {
  this->addNumberSelf(scalar);
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator-=(T scalar) {
  this->subtractNumberSelf(scalar);
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator*=(T scalar) {
  this->multiplyNumberSelf(scalar);
  return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator/=(T scalar) {
  this->divideNumberSelf(scalar);
  return *this;
}

template <class T>
bool Matrix<T>::operator==(const Matrix<T> &other) const {
  return this->equals(other, 1e-9);
}

template <class T>
bool Matrix<T>::operator!=(const Matrix<T> &other) const {
  return !this->equals(other, 1e-9);
}

/* Non-member scalar-matrix operators and stream output */

template <class T>
inline Matrix<T> operator+(T scalar, const Matrix<T> &M) {
  return M + scalar;
}

template <class T>
inline Matrix<T> operator-(T scalar, const Matrix<T> &M) {
  return (-M) + scalar;
}

template <class T>
inline Matrix<T> operator*(T scalar, const Matrix<T> &M) {
  return M * scalar;
}

template <class T>
inline std::ostream& operator<<(std::ostream &os, const Matrix<T> &M) {
  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j < M.columns(); ++j) {
      double val = static_cast<double>(M.get(i, j));
      if (std::abs(val) < EPSILON) val = 0.0;
      os << std::setw(10) << std::fixed << std::setprecision(4) << val << " ";
    }
    os << "\n";
  }
  return os;
}

/**********************************************
 * Algorithm part
 **********************************************/

// Calculate the pairwise interpoint Euclidean distances
// X is data matrix, D is distance matrix
inline void EuclideanDistanceMatrix(const Matrix<double> &X, Matrix<double> &D) {
  if (X.rows() != D.rows() || X.rows() != D.columns()) {
    throw std::invalid_argument("Invalid distance matrix dimension.");
  }

  for (int i = 0; i < D.rows(); i++)
    D.v[i][i] = 0.0;

  for (int i = 0; i < D.rows() - 1; i++) {
    for (int j = i + 1; j < D.columns(); j++) {
      double temp = 0.0;
      for (int k = 0; k < X.columns(); k++) {
        double diff = X.v[i][k] - X.v[j][k];
        temp += diff * diff;
      }
      D.v[i][j] = std::sqrt(temp);
    }
  }

  for (int i = 1; i < D.rows(); i++) {
    for (int j = 0; j < i; j++) {
      D.v[i][j] = D.v[j][i];
    }
  }
}

// Copy all elements of X to Y
inline void ElementCopy(const Matrix<double> &X, Matrix<double> &Y) {
  if (X.rows() != Y.rows() || X.columns() != Y.columns()) {
    throw std::invalid_argument("Invalid matrix dimension.");
  }
  for (int i = 0; i < X.rows(); i++) {
    for (int j = 0; j < X.columns(); j++) {
      Y.v[i][j] = X.v[i][j];
    }
  }
}

// Classical (Torgerson) Metric Multidimensional Scaling
inline Matrix<double> MDS_Classical(const Matrix<double> &D, int dim) {
  if (D.rows() != D.columns()) {
    throw std::invalid_argument("Input distance matrix to MDS is not square.");
  }
  if (dim < 1 || dim > D.rows()) {
    throw std::invalid_argument("Invalid dimension for MDS.");
  }
  smat_t *cD = D.toCMatrix();
  smat_t *cX = smat_mds_classical(cD, dim);
  smat_free(cD);
  if (!cX) {
    throw std::runtime_error("Classical MDS failed.");
  }
  Matrix<double> X(cX);
  smat_free(cX);
  return X;
}

// Multidimensional scaling (MDS)
// This function re-implements Laurens van der Maaten's MDS in his Matlab Toolbox for Dimensionality Reduction
inline Matrix<double> MDS_UCF(const Matrix<double> &D, const Matrix<double> *X0, int dim, int iter) {
  if (D.rows() != D.columns()) {
    throw std::invalid_argument("Input distance matrix to MDS is not square.");
  }
  if (dim < 1) {
    throw std::invalid_argument("Invalid dimension for MDS.");
  }
  if (iter < 1) {
    throw std::invalid_argument("Invalid number of iterations for MDS.");
  }

  Matrix<double> X;
  if (X0 != NULL) {
    if (X0->rows() != D.rows() || X0->columns() != dim) {
      throw std::invalid_argument("Input initialization to MDS has invalid dimension.");
    }
    X = X0->copy();
  } else {
    X = Matrix<double>(D.rows(), dim, "rand");
    double D_mean = D.mean();
    X.addNumberSelf(-0.5);
    X.multiplyNumberSelf(0.1 * D_mean / (1.0 / 3.0 * std::sqrt(static_cast<double>(dim))));
  }

  double lr = 0.05; // learning rate
  double r = 2.0;   // metric
  int n = D.rows(); // number of vectors

  Matrix<double> dh(n, n, 0.0);
  Matrix<double> pmat(n, dim);
  Matrix<double> dhdum(n, 1);
  Matrix<double> dhmat(n > 1 ? n - 1 : 1, dim, 0.0);

  Matrix<int> RP(n, iter, "randperm");
  int i, j;
  double temp;
  int m;

  printf("\n[MDS] Iterations:\n");
  for (int it = 0; it < iter; it++) {
    if (it % 10 == 0)
      printf("\n[MDS] ");
    printf("%3d  ", it + 1);
    for (int rp = 0; rp < n; rp++) {
      m = RP.v[rp][it] - 1;

      for (i = 0; i < n; i++) {
        for (j = 0; j < dim; j++) {
          pmat.v[i][j] = X.v[m][j] - X.v[i][j];
        }
      }

      for (i = 0; i < n; i++) {
        temp = 0;
        for (j = 0; j < dim; j++) {
          temp += std::pow(std::fabs(pmat.v[i][j]), r);
        }
        dhdum.v[i][0] = std::pow(temp, 1.0 / r);
      }

      for (i = 0; i < n; i++) {
        if (i == m) continue;
        dh.v[m][i] = dhdum.v[i][0];
        dh.v[i][m] = dhdum.v[i][0];
      }

      for (i = 0; i < n - 1; i++) {
        int ii = (i >= m) ? (i + 1) : i;
        double dist = dhdum.v[ii][0];
        temp = (dist > 1e-12) ? (lr * (dist - D.v[ii][m]) * std::pow(dist, 1.0 - r)) : 0.0;
        for (j = 0; j < dim; j++) {
          dhmat.v[i][j] = temp;
        }
      }

      for (i = 0; i < n - 1; i++) {
        int ii = (i >= m) ? (i + 1) : i;
        for (j = 0; j < dim; j++) {
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
inline Matrix<double> MDS_SMACOF(const Matrix<double> &D, const Matrix<double> *X0, int dim, int iter) {
  if (D.rows() != D.columns()) {
    throw std::invalid_argument("Input distance matrix to MDS is not square.");
  }
  if (dim < 1) {
    throw std::invalid_argument("Invalid dimension for MDS.");
  }
  if (iter < 1) {
    throw std::invalid_argument("Invalid number of iterations for MDS.");
  }

  Matrix<double> X;
  if (X0 != NULL) {
    if (X0->rows() != D.rows() || X0->columns() != dim) {
      throw std::invalid_argument("Input initialization to MDS has invalid dimension.");
    }
    X = X0->copy();
  } else {
    X = Matrix<double>(D.rows(), dim, "rand");
    double D_mean = D.mean();
    X.addNumberSelf(-0.5);
    X.multiplyNumberSelf(0.1 * D_mean / (1.0 / 3.0 * std::sqrt(static_cast<double>(dim))));
  }

  Matrix<double> Z = X.copy();
  Matrix<double> D_(D.rows(), D.columns());
  Matrix<double> B(D.rows(), D.columns());
  int i, j, k;
  double temp;

  EuclideanDistanceMatrix(X, D_);

  printf("\n[MDS] Iterations:\n");
  for (int it = 0; it < iter; it++) {
    if (it % 10 == 0)
      printf("\n[MDS] ");
    printf("%3d  ", it + 1);

    for (i = 0; i < D.rows(); i++) {
      for (j = 0; j < D.columns(); j++) {
        if (i == j || std::fabs(D_.v[i][j]) < EPSILON) {
          B.v[i][j] = 0.0;
        } else {
          B.v[i][j] = -D.v[i][j] / D_.v[i][j];
        }
      }
    }

    for (j = 0; j < D.columns(); j++) {
      temp = 0;
      for (i = 0; i < D.rows(); i++) {
        temp += B.v[i][j];
      }
      B.v[j][j] = -temp;
    }

    for (i = 0; i < X.rows(); i++) {
      for (j = 0; j < X.columns(); j++) {
        temp = 0;
        for (k = 0; k < B.columns(); k++) {
          temp += (B.v[i][k] * Z.v[k][j]);
        }
        X.v[i][j] = temp / static_cast<double>(D.rows());
      }
    }

    EuclideanDistanceMatrix(X, D_);
    Z = X;
  }

  printf("\n");
  return X;
}

} // namespace smat

#endif /* __cplusplus */

#endif /* SIMPLE_MATRIX_H */
