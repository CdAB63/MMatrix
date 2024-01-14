arithmetic
multiplyPerElement: mat
	"multiply two matrixes element wise"
	| r |
	self assert: [ nbRows = mat nbRows ].
	self assert: [ nbColumns = mat nbColumns ].
	r := self class newRows: nbRows columns: nbColumns.
	r fromContents: (self asArray with: mat array asArray collect: [:a :b |
		a * b]).
	^r