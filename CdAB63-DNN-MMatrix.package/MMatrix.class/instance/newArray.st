initialization
newArray
	"Create an array used to keep the matrix content"
	^ FFIExternalArray
			newType: 'double' size: nbColumns * nbRows