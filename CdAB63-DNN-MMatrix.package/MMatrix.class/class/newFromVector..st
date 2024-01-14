instance creation
newFromVector: array
	"Create a Nx1 matrix from an array of numbers"
	^ self basicNew
				initializeRows: array size columns: 1;
				fromContents: array;
				yourself