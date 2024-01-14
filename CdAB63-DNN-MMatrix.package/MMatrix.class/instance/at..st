accessing
at: aPoint
	"access element of the matrix"
	^ array at: ((aPoint x - 1) * nbColumns + (aPoint y - 1)) + 1