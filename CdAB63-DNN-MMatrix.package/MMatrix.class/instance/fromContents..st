initialization
fromContents: contents
	"Initialize the matrix with linear content"
	self assert: [ contents size = (nbColumns * nbRows) ]
		  description: 'Size mismatch'.
	contents doWithIndex: [ :v :i | array at: i put: v ]