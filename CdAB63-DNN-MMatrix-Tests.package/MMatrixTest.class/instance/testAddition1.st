tests
testAddition1
	| m1 m2 |
	m1 := MMatrix newFromArrays: #(#(1 2 3) #(4 5 6)).
	m2 := MMatrix newFromArrays: #(#(4 5 6) #(1 2 3)).
	self assert: (m1 + m2) asStructuredArray equals: #(#(5.0 7.0 9.0) #(5.0 7.0 9.0))