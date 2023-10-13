def flatten(l):

    res = []

    if not hasattr(l, '__iter__'):
        return [l] #parentesi quadre perchè extend vuole un iterabile e così si crea una lista di un singolo elemento

    for item in l:
        res.extend(flatten(item))

    return res

if __name__ == '__main__':
    print(flatten([[1,2],[5,7],5,3]))