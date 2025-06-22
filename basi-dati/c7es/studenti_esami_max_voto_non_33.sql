SELECT e.matr, MAX(e.voto) AS voto_max
FROM e
WHERE e.voto <> 33
GROUP BY e.matr
HAVING COUNT(*) > 1
ORDER BY voto_max ASC
