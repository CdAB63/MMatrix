instance creation
newFromArrays: arrays
	"Create a matrix from an array containing the structured values of the matris
	For instance: #(#(1 2 3) #(4 5 6))"
	^ self basicNew
				initializeRows: arrays size columns: arrays first size;
				fromContents: (arrays flatCollect: #yourself);
				yourself