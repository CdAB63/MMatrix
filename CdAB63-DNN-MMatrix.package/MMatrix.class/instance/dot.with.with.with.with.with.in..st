as yet unclassified
dot: array1 with: m1_nb_rows with: m1_nb_columns with: array2
            with: m2_nb_rows with: m2_nb_columns
            in: res
	"Invoke the C library to perform the operator"
	^self
		ffiCall: #(void dot(void *array1, int m1_nb_rows, int m1_nb_columns,
		                    void *array2, int m2_nb_rows, int m2_nb_columns,
		                    void *res)) module: 'libmatrix.so'