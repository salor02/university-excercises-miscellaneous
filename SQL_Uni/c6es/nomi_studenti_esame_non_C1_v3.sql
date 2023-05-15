SELECT s.snome
FROM s
WHERE NOT EXISTS(
	SELECT *
	FROM e
	WHERE e.matr = s.matr
	AND e.cc = 'C1')