tests
testStretching
	| m |
	m := (MMatrix newFromVector: #(1 2 3 4)) stretchToColumns: 5.
	self assert: m nbRows equals: 4.
	self assert: m nbColumns equals: 5.
	self assert: (m atRow: 1) equals: #(1 1 1 1 1).
	self assert: (m atRow: 3) equals: #(3 3 3 3 3).