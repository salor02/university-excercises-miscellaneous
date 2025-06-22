SELECT t1.cd, t1.cc
FROM DOC_COR_ESAMI AS t1
WHERE t1.num_esami >= ALL(
	SELECT t2.num_esami
	FROM DOC_COR_ESAMI AS t2
	WHERE t2.cd = t1.cd)