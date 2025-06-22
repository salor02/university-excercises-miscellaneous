SELECT matr
FROM e
GROUP BY matr, voto
HAVING COUNT(*) >= 2