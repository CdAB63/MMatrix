arithmetic
sub: anotherMatrix
	"subtract a matrix from another matrix"
	| result resultArray |
	
	nbColumns = anotherMatrix nbColumns ifFalse: [ self error: 'dimensions do not conform' ].
	nbRows = anotherMatrix nbRows ifFalse: [ self error: 'dimensions do not conform' ].
	self assert: [ nbRows * nbColumns = array size ].
	self assert: [ anotherMatrix nbRows * anotherMatrix nbColumns = anotherMatrix size ].
	
	resultArray := ByteArray new: (nbRows * nbColumns * 8).
	
	self
		sub: self getHandle with: nbRows with: nbColumns 
		                    with: anotherMatrix getHandle 
		                    in: resultArray.
	
	result := MMatrix newHandle: resultArray rows: nbRows columns: nbColumns.
	
	^ result