accessing
asStructuredArray
	"Return a structured array that describe the matrix"
	^ (1 to: nbRows) collect: [ :i | self atRow: i ] as: Array