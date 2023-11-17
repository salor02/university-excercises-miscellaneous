from functools import reduce

str = "ciahsdfjhsdbfjabfjsbadlfjb"

#non so cosa significhi
res = reduce(lambda x,y: x | {y:x.get(y,0)+1}, str, {})

print(res)