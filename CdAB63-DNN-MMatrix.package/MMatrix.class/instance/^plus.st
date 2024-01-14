arithmetic
+ matrixOrVector
	"Add either a matrix or a vector to the receiver"
	| m |
	((nbRows = matrixOrVector nbRows) and: [ nbColumns = matrixOrVector nbColumns ])
		ifTrue: [ ^self add: matrixOrVector ].
	matrixOrVector nbColumns ~= 1 ifTrue: [ self error: 'not a n * 1 vector' ].
	m := matrixOrVector stretchToColumns: nbColumns.
	^ self + m