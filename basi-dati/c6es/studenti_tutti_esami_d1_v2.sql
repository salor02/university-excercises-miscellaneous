SELECT s.*
FROM s
WHERE NOT EXISTS(
	SELECT e.cc
	FROM e JOIN c ON e.cc = c.cc
	WHERE c.cd = 'D1'
	EXCEPT
	SELECT e.cc
	FROM e
	WHERE e.matr = s.matr)
	
	