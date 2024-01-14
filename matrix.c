#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Multiply matrixes (dot product)
void dot(double *m1, int m1_nb_rows, int m1_nb_columns, 
	 double *m2, int m2_nb_rows, int m2_nb_columns,
	 double *res) {
	int col, row, k;
	for (col=0; col < m2_nb_columns; col++) {
		for (row=0; row < m1_nb_rows; row++) {
			double tmp = 0;
			for (k=0; k < m2_nb_rows; k++)
				tmp += m1[row*m1_nb_columns+k] * m2[k*m2_nb_columns+col];
			res[row*m2_nb_columns+col] = tmp;
		}
	}
}

// Subtract matrixes
void sub(double *m1, int nb_rows, int nb_columns,
	 double *m2, double *res) {
	int col, row;
	for (col=0; col < nb_columns; col++) {
		for (row=0; row < nb_rows; row++) {
			res[row * nb_columns + col ] = 
				m1[row * nb_columns + col ] - m2[row * nb_columns + col];
		}
	}
}

// Add matrixes
void add(double *m1, int nb_rows, int nb_columns,
	 double *m2, double *res) {
	int col, row;
	for (col=0; col < nb_columns; col++) {
		for (row=0; row < nb_rows; row++) {
			res[row * nb_columns + col] =
				m1[row * nb_columns + col] + m2[row * nb_columns + col];
		}
	}
}

// ****************************************************************************
// Determinant calculation

// Function to get the minor of a matrix excluding the specified row and column
void getMinor(double *src, double *dest, int row, int col, int size) {
    int i = 0, j = 0;

    for (int r = 0; r < size; r++) {
        if (r != row) {
            for (int c = 0; c < size; c++) {
                if (c != col) {
                    dest[i * (size - 1) + j] = src[r * size + c];
                    j++;
                }
            }
            j = 0;
            i++;
        }
    }
}

// Recursive function to calculate the determinant of a matrix
double determinant(double *matrix, int size) {
    if (size == 1) {
        return matrix[0];
    }

    if (size == 2) {
        return matrix[0] * matrix[3] - matrix[1] * matrix[2];
    }

    double det = 0.0;
    double *minor = (double *) malloc((size - 1) * (size - 1) * sizeof(double));

    for (int i = 0; i < size; i++) {
        getMinor(matrix, minor, 0, i, size);
        det += ((i % 2 == 1) ? -1.0 : 1.0) * matrix[i] * determinant(minor, size - 1);
    }

    free(minor);

    return det;
}

// Wrapper function to calculate determinant from a unidimensional array
double det(double *matrix, int nbRows) {
    return determinant(matrix, nbRows);
}

// ****************************************************************************
// Inverse matrix

// Function to perform LU decomposition with partial pivoting
void luDecomposition(double *matrix, int size, int *permutation) {
    for (int i = 0; i < size; i++) {
        permutation[i] = i;
    }

    for (int k = 0; k < size - 1; k++) {
        double max = 0.0;
        int pivot = -1;

        for (int i = k; i < size; i++) {
            double val = fabs(matrix[i * size + k]);
            if (val > max) {
                max = val;
                pivot = i;
            }
        }

        if (pivot == -1) {
            // Matrix is singular
            return;
        }

        // Swap rows
        if (pivot != k) {
            int temp = permutation[k];
            permutation[k] = permutation[pivot];
            permutation[pivot] = temp;

            for (int j = 0; j < size; j++) {
                double tempVal = matrix[k * size + j];
                matrix[k * size + j] = matrix[pivot * size + j];
                matrix[pivot * size + j] = tempVal;
            }
        }

        for (int i = k + 1; i < size; i++) {
            matrix[i * size + k] /= matrix[k * size + k];

            for (int j = k + 1; j < size; j++) {
                matrix[i * size + j] -= matrix[i * size + k] * matrix[k * size + j];
            }
        }
    }
}

// Function to solve a linear system Ax = B using LU decomposition
void solveLinearSystem(double *LU, int *permutation, double *B, double *x, int size) {
    // Forward substitution (Ly = B)
    for (int i = 0; i < size; i++) {
        x[i] = B[permutation[i]];

        for (int j = 0; j < i; j++) {
            x[i] -= LU[i * size + j] * x[j];
        }
    }

    // Backward substitution (Ux = y)
    for (int i = size - 1; i >= 0; i--) {
        for (int j = i + 1; j < size; j++) {
            x[i] -= LU[i * size + j] * x[j];
        }

        x[i] /= LU[i * size + i];
    }
}

// Function to calculate the inverse of a square matrix
void inverse(double *matrix, int nbRows, double *inverseMatrix) {
    if (nbRows <= 0 || matrix == NULL || inverseMatrix == NULL) {
        // Invalid input
        return;
    }

    int size = nbRows;

    double *LU = malloc(size * size * sizeof(double));
    int *permutation = malloc(size * sizeof(int));
    double *B = malloc(size * sizeof(double));
    double *x = malloc(size * sizeof(double));

    // Initialize inverseMatrix as the identity matrix
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            inverseMatrix[i * size + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Perform LU decomposition on the original matrix
    luDecomposition(matrix, size, permutation);

    // Solve linear systems for each column of the identity matrix
    for (int col = 0; col < size; col++) {
        // Extract the column of the identity matrix as the right-hand side vector B
        for (int row = 0; row < size; row++) {
            B[row] = inverseMatrix[row * size + col];
        }

        // Solve the linear system LUx = B for x
        solveLinearSystem(matrix, permutation, B, x, size);

        // Copy the solution x into the corresponding column of the inverse matrix
        for (int row = 0; row < size; row++) {
            inverseMatrix[row * size + col] = x[row];
        }
    }

    free(LU);
    free(permutation);
    free(B);
    free(x);
}

