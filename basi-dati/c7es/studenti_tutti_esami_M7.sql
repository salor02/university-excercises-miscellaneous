SELECT s.*
FROM s
WHERE s.matr <> 'M7'
AND NOT EXISTS( --non esiste nessun esame dello studente M7
	SELECT e.*
	FROM e
	WHERE e.matr = 'M7'
	AND e.cc NOT IN( --che non sia tra gli esami sostenuti dallo studente corrente
		SELECT e.cc
		FROM e
		WHERE e.matr = s.matr)
	)
	