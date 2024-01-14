tests
testSub
	| m1 m2 |
	
	m1 := MMatrix newFromArrays: #(#(1 2 3 4) #(5 6 7 8)).
	m2 := MMatrix newFromArrays: #(#(4 2 1 3) #(7 6 8 5)).
	self assert: (m1 - m2) asStructuredArray equals: #(#(-3 0 2 1) #(-2 0 -1 3))