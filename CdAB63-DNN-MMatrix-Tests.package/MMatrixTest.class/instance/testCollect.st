tests
testCollect
	| m expectedMatrix |
	m := MMatrix newFromArrays: #(#(1 2 3) #(4 5 6)).
	expectedMatrix := MMatrix newFromArrays: #(#(2 3 4) #(5 6 7)).
	self assert: (m collect: [ :v | v + 1]) asStructuredArray
		  equals: expectedMatrix asStructuredArray.