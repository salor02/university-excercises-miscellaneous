str = "ciahsdfjhsdbfjabfjsbadlfjb"

res = map(lambda letter: letter.upper(), str)
res = filter(lambda letter:letter not in ['A','E','I','O','U'], res)

print(list(res))