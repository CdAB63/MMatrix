tests
testInverse
	| m i |
	m := MMatrix newFromArrays:  #(#(1.0 0.0) #(0.0 1.0)).
	i := m inverse.
	self assert: i asStructuredArray equals: #(#(1.0 0.0) #(0.0 1.0)).
	m := MMatrix newFromArrays: #(#(0.5 0.0) #(0.0 0.5)).
	i := m inverse.
	self assert: i asStructuredArray equals: #(#(2.0 0.0) #(0.0 2.0)).