tests
testAt
	| m |
	
	m := MMatrix newFromArrays: #(#(1 2) #(3 4)).
	self assert: (m at: 1 @ 1) equals: 1.0.
	self assert: (m at: 1 @ 2) equals: 2.0.
	self assert: (m at: 2 @ 1) equals: 3.0.
	self assert: (m at: 2 @ 2) equals: 4.0.