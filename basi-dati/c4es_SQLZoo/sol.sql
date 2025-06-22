--2
select yr
from movie
where title = 'Citizen Kane'

--4
select id
from actor
where name = 'Glenn Close'

--5
select id
from movie
where title = 'Casablanca'

--7
select actor.name
from casting
join movie on casting.movieid = movie.id
join actor on casting.actorid = actor.id
where movie.title = 'Alien'

--8
select movie.title
from casting
join movie on casting.movieid = movie.id
join actor on casting.actorid = actor.id
where actor.name = 'Harrison Ford'

--9
select movie.title
from casting
join movie on casting.movieid = movie.id
join actor on casting.actorid = actor.id
where actor.name = 'Harrison Ford'
and casting.ord <> 1

--10
select movie.title, actor.name
from casting
join movie on casting.movieid = movie.id
join actor on casting.actorid = actor.id
where casting.ord = 1
and movie.yr = 1962

--15
select actor.name
from casting 
join actor on casting.actorid = actor.id
join movie on casting.movieid = movie.id
where actor.name <> 'Art Garfunkel'
and movieid in(
select casting.movieid
from casting
join actor on casting.actorid = actor.id
where actor.name = 'Art Garfunkel')

