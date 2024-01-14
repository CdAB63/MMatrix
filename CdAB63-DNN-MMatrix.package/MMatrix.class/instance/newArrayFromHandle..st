initialization
newArrayFromHandle: aHandle
	"Create an external array using a handle"
	^ FFIExternalArray
			fromHandle: aHandle type: 'double' size: nbColumns * nbRows