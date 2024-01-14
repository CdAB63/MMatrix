as yet unclassified
dot: anotherMatrix
	"dot product between matrixes"
	| result resultArray |
	nbColumns = anotherMatrix nbRows ifFalse: [ self error: 'dimensions do not conform' ].
	self assert: [ nbRows * nbColumns = array size ].
	self assert: [ anotherMatrix nbRows * anotherMatrix nbColumns = anotherMatrix size ].
	resultArray := ByteArray new: (nbRows * anotherMatrix nbColumns * 8).
	self
		dot: self getHandle with: nbRows with: nbColumns with: anotherMatrix getHandle 
		with: anotherMatrix nbRows with: anotherMatrix nbColumns in: resultArray.
	result := self class newHandle: resultArray rows: nbRows columns: anotherMatrix nbColumns.
	^ result