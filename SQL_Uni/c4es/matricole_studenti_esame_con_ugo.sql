select e.matr
from e
where e.cc in (
	select e.cc
	from e 
	join s on e.matr = s.matr
	where s.snome = 'Ugo Rossi'
)
		
