rubrica = {}

while True:
    scelta = input("Operazione (print, ins, canc): ")

    if scelta == 'ins':
        num = input("Numero di telefono: ")
        nome = input("Nome: ")
        rubrica[num] = nome
    elif scelta == 'canc':
        num = input("Numero di telefono da eliminare: ")
        if num in rubrica:
            del rubrica[num]
    elif scelta == 'print':
        for item in rubrica.items():
            print(item)
    else: 
        break