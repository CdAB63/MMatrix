as yet unclassified
stretchToColumns: nbOfColumns
	"stretch vertical vector in columns"
	| content result |
	content := OrderedCollection new.
	1 to: nbRows do: [ :row |
		1 to: nbOfColumns do: [ :columns |
			content add: (self at: row @ 1) ] ].
	result := self class newRows: nbRows columns: nbOfColumns.
	result fromContents: content.
	^ result