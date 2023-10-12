from urllib import request
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

#url = "https://www.gutenberg.org/cache/epub/71794/pg71794.txt"
#response = request.urlopen(url)
#text = response.read().decode('utf8')
text = 'cat mouse dog is planet farm wheat rice red blue green Pokemon'

#tokenizzazione
print("Tokenizing...")
tokens = nltk.word_tokenize(text)
print("Done!")

#POS tagging
print("\nPOS tagging...")
tokens = nltk.pos_tag(tokens)
print("Done!")

result = set()

print("\nDeleting stopwords and stemming...")
for t in tokens:
    if 'NN' in t[1]:
        if not t[0] in stopwords.words('english'):
            #result.add(porter.stem(t[0]))
            result.add(t[0])
print("Done!")

print("\nRESULT:\n")
print(result)

#word disambiguation
for word in result:
    for meaning in wn.synsets(word, wn.NOUN):
        print(meaning)
    