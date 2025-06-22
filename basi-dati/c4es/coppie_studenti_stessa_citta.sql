select a.matr, b.matr
from s as a,s as b
where a.citta = b.citta
and a.matr < b.matr
