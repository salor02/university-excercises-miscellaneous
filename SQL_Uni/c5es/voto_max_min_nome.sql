SELECT s.matr, s.snome, MAX(voto) AS voto_max, MIN(voto) AS voto_min
FROM e join s on e.matr = s.matr
GROUP BY s.matr