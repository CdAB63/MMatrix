tests
testMultiplicationOnMatrix
	| x |
	x := MMatrix newFromArrays: #(#(1 2 3 4) #(10 20 30 40)).
	self assert: (x * 5) asStructuredArray equals: #(#(5.0 10.0 15.0 20.0)
																	#(50.0 100.0 150.0 200.0))