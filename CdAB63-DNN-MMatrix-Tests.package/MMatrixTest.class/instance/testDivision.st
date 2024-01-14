tests
testDivision 
	| m |
	m := MMatrix newFromArrays: #(#(1 2 3) #(4 5 6)).
	self assert: (m / 2) asStructuredArray equals: #(#(0.5 1.0 1.5) #(2.0 2.5 3.0)).