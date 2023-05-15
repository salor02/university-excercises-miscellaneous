SELECT s.snome
FROM s
WHERE s.matr <> ALL(
	SELECT e.matr
	FROM e
	WHERE e.cc = 'C1')