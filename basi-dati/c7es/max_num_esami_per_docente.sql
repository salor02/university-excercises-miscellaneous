SELECT c1.cd, c1.cc
FROM c AS c1 JOIN e AS e1 ON c1.cc = e1.cc
GROUP BY c1.cd, c1.cc
HAVING COUNT(*) >= ALL( --confronta con il numero di tutti gli esami di quel docente
	SELECT COUNT(*)
	FROM c AS c2 JOIN e AS e2 ON c2.cc = e2.cc
	AND c2.cd = c1.cd
	GROUP BY c2.cd, c2.cc 
)



