CREATE VIEW DOC_COR_ESAMI AS
SELECT c.cd, c.cc, COUNT(*)
FROM c JOIN e ON c.cc = e.cc
GROUP BY c.cd, c.cc