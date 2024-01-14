instance creation
newRows: numRows columns: numColumns
	"Create a matrix with a given shape"
	^ self basicNew
				initializeRows: numRows columns: numColumns;
				yourself