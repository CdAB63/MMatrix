point functions
transposed 
	"Transpose the matrix"
	| result |
	
	result := self class newRows: nbColumns columns: nbRows.
	1 to: nbRows do: [ :row |
		1 to: nbColumns do: [ :column |
			result at: column @ row put: (self at: row @ column) ] ].
	^ result