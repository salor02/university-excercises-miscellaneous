SELECT s.matr
FROM s JOIN e ON s.matr = e.matr
WHERE s.matr <> 'M2'
AND (e.cc,e.data) IN( --deve essere tra gli esami sostenuti da M2
	SELECT e.cc, e.data
	FROM e
	WHERE e.matr = 'M2')
	