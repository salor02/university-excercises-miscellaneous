SELECT s.snome
FROM s
WHERE s.matr NOT IN(
	SELECT e.matr
	FROM e
	WHERE e.cc = 'C1')