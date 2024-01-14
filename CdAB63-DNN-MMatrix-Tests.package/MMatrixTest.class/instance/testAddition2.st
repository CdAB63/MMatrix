tests
testAddition2
	| m |
	
	m := MMatrix newFromArrays: #(#(1 2 3) #(4 5 6)).
	self assert: (m + m) asStructuredArray equals: #(#(2.0 4.0 6.0) #(8.0 10.0 12.0))