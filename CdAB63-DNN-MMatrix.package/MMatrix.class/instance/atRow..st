accessing
atRow: rowNumber
	"Return a particular row"
	(rowNumber between: 1 and: rowNumber) ifFalse: [ self error: 'index out of range' ].
	^ (1 to: nbColumns) collect: [ :x | self at: rowNumber @ x ]