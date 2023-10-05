import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

wnl = nltk.WordNetLemmatizer()

text = input("Text to process: ")

#tokenizzazione
tokens = nltk.word_tokenize(text)

#lemmatization
print("\nLemmatize...")
for t in tokens:
    if not t in stopwords.words('english'):
        print(wnl.lemmatize(t))

#stemmer
porter = PorterStemmer()
print("\nStemming...")
print([porter.stem(t) for t in tokens])

#POS tagging
print("\nPOS tagging...")
print(nltk.pos_tag(tokens))