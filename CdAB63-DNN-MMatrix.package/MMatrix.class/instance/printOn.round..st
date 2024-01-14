printing
printOn: aStream round: nbDecimals
	"print the receiver matrix into a stream. All numerical values are
	truncated to a fixed number of decimals"
	aStream nextPutAll: '('.
	(1 to: nbRows) do: [ :r |
		(self atRow: r) do: [ 
			:each | aStream nextPutAll: (each round: nbDecimals) printString ] 
					separatedBy: [ aStream space  ]] separatedBy: [ aStream lf ].
	aStream nextPutAll: ')'.