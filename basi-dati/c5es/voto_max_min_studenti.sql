SELECT matr, MAX(voto) AS voto_max, MIN(voto) AS voto_min
FROM e
WHERE cc <> 'C2'
GROUP BY matr