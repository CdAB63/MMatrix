tests
testCreation
	| m |
	m := MMatrix newFromArrays: #(#(1 2) #(3 4)).
	self assert: m asArray equals: #(1.0 2.0 3.0 4.0)