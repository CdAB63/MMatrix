enumerating
collect: aOneArgBlock
	"Return a new matrix for which each matrix element is
	transformed using the provided block"
	| result |
	
	result := self class newRows: nbRows columns: nbColumns.
	1 to: nbRows do: [ :y |
		1 to: nbColumns do: [ :x |
			result at: y @ x put: (aOneArgBlock value: (self at: y @ x)) ] ].
	^ result