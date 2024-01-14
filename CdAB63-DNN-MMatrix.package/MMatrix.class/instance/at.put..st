accessing
at: aPoint put: aNumber

	array at: ((aPoint x - 1) * nbColumns + (aPoint y - 1)) + 1
		   put: aNumber asFloat