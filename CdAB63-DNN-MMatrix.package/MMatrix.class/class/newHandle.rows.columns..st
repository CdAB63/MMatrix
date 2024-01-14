instance creation
newHandle: aHandle rows: numRows columns: numColumns
	"Create a matrix with a provided content"
	^ self basicNew
				initializeHandle: aHandle rows: numRows columns: numColumns;
				yourself