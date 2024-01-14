tests
testMax

	| m |
	m := MMatrix newFromArrays: #(#(1 2) #(3 4)).
	self assert: m max equals: 4.0.