SELECT s.*
FROM s JOIN e ON e.matr = s.matr
--seleziona una riga se il codice corso è uno di quelli tenuti dal docente D1
WHERE e.cc IN(
	SELECT c.cc
	FROM c
	WHERE c.cd = 'D1')
--raggruppa gli esami per studente (matricola)
GROUP BY s.matr
--se il numero di esami nel gruppo combacia col numero di esami 
HAVING COUNT(*) = (
	SELECT COUNT(*)
	FROM c
	WHERE c.cd= 'D1')
	
	
