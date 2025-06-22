SELECT e.cc, AVG(e.voto)
FROM e
GROUP BY cc
HAVING COUNT(*) >= 2