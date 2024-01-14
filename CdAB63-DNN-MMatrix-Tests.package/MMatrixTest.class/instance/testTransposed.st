tests
testTransposed

	| m expectedResult |
	
	m := MMatrix newFromArrays: #(#(1 2 3 4) #(5 6 7 8)).
	expectedResult := MMatrix newFromArrays: #(#(1 5) #(2 6) #(3 7) #(4 8)).
	self assert: m transposed asStructuredArray equals: expectedResult asStructuredArray.
	m := MMatrix newFromVector: #(1 2 3 4).
	self assert: m transposed asStructuredArray equals: #(#(1 2 3 4)).