tests
testAtPut
	| m |
	m := MMatrix newFromArrays: #(#(1 2) #(3 4)).
	m at: 2 @ 1 put: 10.0.
	self assert: (m at: 2 @ 1) equals: 10.0.
	self assert: m asStructuredArray equals: #(#(1.0 2.0) #(10.0 4.0))