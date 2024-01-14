arithmetic
det
	"Calculate the determinant"
	| d |
	nbRows = 0 ifTrue: [ self error: 'dimensions do not conform' ].
	nbRows = nbColumns ifFalse: [ self error: 'dimensions do not conform' ].
	d := self det: self getHandle dim: nbRows.
	^ d.