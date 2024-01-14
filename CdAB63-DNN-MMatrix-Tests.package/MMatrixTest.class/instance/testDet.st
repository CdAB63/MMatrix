tests
testDet
	| m d |
	
	m := MMatrix newFromArrays: #(#(2 0 0) #(0 2 0) #(0 0 2)).
	d := m det.
	self assert: d equals: 8.0.
	m := MMatrix newFromArrays: #(#(1 2) #(3 4)).
	d := m det.
	self assert: d equals: -2.
	m := MMatrix newFromArrays: #(#(1 2 3) #(4 5 6) #(7 8 9)).
	d := m det.
	self assert: d equals: 0.0.
	m := MMatrix newFromArrays: #(#(1 0 3) #(0 5 6) #(7 8 0)).
	d := m det.
	self assert: d equals: -153.0.