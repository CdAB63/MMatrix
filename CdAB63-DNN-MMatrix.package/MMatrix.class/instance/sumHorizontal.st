arithmetic
sumHorizontal
	"Horizontal summing"
	| result sum |
	
	result := self class newRows: nbRows columns: 1.
	1 to: nbRows do: [  :y |
		sum := 0.
		1 to: nbColumns do: [ :x |
			sum := sum + (self at: y @ x) ].
		result at: y @ 1 put: sum ].
	^ result