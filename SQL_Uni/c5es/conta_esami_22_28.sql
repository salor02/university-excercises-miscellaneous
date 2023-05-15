SELECT voto, COUNT(*)
FROM e
GROUP BY voto
HAVING voto >= 22 AND voto <= 28