class Menu:
    def __init__(self, name, ops):
        self.name = name
        self.ops = ops
    
    #display delle varie opzioni disponibili
    def display(self):
        print('*'*40+'\n' + self.name)
        for op in self.ops.items():
            print(f"{op[0]}\t-\t{op[1][0]}")
        print('Digitare 0 per uscire')
        print('*'*40)
    
    #prompt per inserire la scelta e successiva esecuzione dell'azione associata
    def select(self):
        try:
            choice = int(input("Selezionare opzione desiderata: "))
        except ValueError:
            return 'REPEAT'
        
        if choice == 0:
            return 'EXIT' #codice per comunicare al main di uscire
        elif choice < 1 or choice > len(self.ops):
            return 'REPEAT' #codice per comunicare al main di riproporre la scelta
        else:
            print('*'*40)
            return self.ops[choice][1]()