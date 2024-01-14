arithmetic
add: aMatrix
	"Add two matrixes"
	| result resultArray |
	nbColumns = aMatrix nbColumns ifFalse: [ self error: 'dimensions do not conform' ].
	nbRows = aMatrix nbRows ifFalse: [ self error: 'dimensions do not conform' ].
	resultArray := ByteArray new: (nbRows * aMatrix nbColumns * 8).
	self assert: [ nbRows * nbColumns = array size ].
	self assert: [ aMatrix nbRows * aMatrix nbColumns = aMatrix size ].
	self assert: [ nbRows * aMatrix nbColumns * 8 = resultArray size ].
	self add: self getHandle with: nbRows with: nbColumns with: aMatrix getHandle in: resultArray.
	result := MMatrix newHandle: resultArray rows: nbRows columns: nbColumns.
	^ result