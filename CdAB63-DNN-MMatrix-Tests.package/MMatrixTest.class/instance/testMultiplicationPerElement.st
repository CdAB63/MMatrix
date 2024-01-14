tests
testMultiplicationPerElement
	| v1 v2 expectedVector |
	v1 := MMatrix newFromVector: #(1 2 3).
	v2 := MMatrix newFromVector: #(10 20 30).
	expectedVector :=MMatrix newFromVector: #(10 40 90).
	self assert: (v1 multiplyPerElement: v2) asArray equals: expectedVector asArray.