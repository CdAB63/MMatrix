information
random: randomNumberGenerator
	"Fill the matrix with random values"
	self fromContents: ((1 to: nbRows * nbColumns) collect: [ :vv |
		randomNumberGenerator next ])