initialization
initializeHandle: aHandle rows: numRows columns: numColumns
	"Initialize the matrix"
	self initialize.
	nbRows := numRows.
	nbColumns := numColumns.
	array := self newArrayFromHandle: aHandle