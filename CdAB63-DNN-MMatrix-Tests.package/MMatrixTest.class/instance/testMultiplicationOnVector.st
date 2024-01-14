tests
testMultiplicationOnVector
	| x |
	x := MMatrix newFromVector: #(1 2 3 4).
	self assert: (x * 5) asStructuredArray equals: #(#(5.0) #(10.0) #(15.0) #(20.0))