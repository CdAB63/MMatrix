operations
inverse
	"calculate the inverse of the matrix"
	| inverseMatrix result |
	
	nbRows = 0 ifTrue: [ self error: 'dimensions do not conform' ].
	nbRows = nbColumns ifFalse: [ self error: 'dimensions do not conform' ].
	
	inverseMatrix := ByteArray new: (nbRows * nbRows * 8).

	self inverse: self getHandle with: nbRows in: inverseMatrix.
	result := self class newHandle: inverseMatrix rows: nbRows columns: nbColumns.
	^ result