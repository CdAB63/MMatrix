arithmetic
inverse: m1 with: dim in: ret

	Transcript 
		cr;
		show: 'DIM: ';
		show: dim;
		cr;
		show: m1 class;
		show: ' size: '; show: m1 size; cr;
		show: ret class; show: ' size: '; show: ret size;
		cr.

	^ self ffiCall: #(void inverse(double *m1, int dim, double *ret)) module: 'libmatrix.so'.