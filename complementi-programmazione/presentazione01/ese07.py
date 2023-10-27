text = input("Inserire testo: ")

#rimuove spazi
text = text.replace(' ','')
letters_count = len(text)

frequency = {}

for char in text:
    frequency[char] = frequency.get(char, 0) + 1

max = None

for i in range (0, len(frequency)):
    for item in frequency.items():
        if item[1] > max:
            max = item[1]
    print(frequency[max])        
