tests
testVectorCreation
	| v |
	v := MMatrix newFromVector: #(1 2 3).
	self assert: v nbColumns equals: 1.
	self assert: v nbRows equals: 3.
	self assert: v asStructuredArray equals: #(#(1) #(2) #(3))