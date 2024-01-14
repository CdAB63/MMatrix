tests
testSumHorizontal
	| m expectedResult |
	m := MMatrix newFromArrays: #(#(1 2) #(3 4) #(5 6)).
	expectedResult := MMatrix newFromArrays: #(#(3.0) #(7.0) #(11.0)).
	self assert: m sumHorizontal asStructuredArray equals: expectedResult asStructuredArray.