SELECT c.cc, c.cnome, COUNT(*)
FROM e join c on e.cc = c.cc
GROUP BY c.cc