arithmetic
add: m1 with: nb_rows with: nb_columns with: m2 in: res
	^ self ffiCall: #(void add(double *m1, int nb_rows, int nb_columns, double *m2, double *res)) module: 'libmatrix.so'