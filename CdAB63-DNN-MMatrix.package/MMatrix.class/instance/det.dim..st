as yet unclassified
det: m1 dim: dim
	"Calculate the determinant of the m1 matrix"
	^ self ffiCall: #(double det(double *m1, int dim)) module: 'libmatrix.so'