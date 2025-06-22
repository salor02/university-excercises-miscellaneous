select s.snome, c.cd
from e 
	join c on e.cc = c.cc
	join s on e.matr = s.matr
where e.voto > 24
		
