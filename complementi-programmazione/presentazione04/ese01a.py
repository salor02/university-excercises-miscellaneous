lista = [1,2,3,4,5]

res = [x**2 for x in lista]
resd = {x:x**2 for x in lista}

resd = filter(lambda x: (resd[x]>1), resd)

print(resd.keys())