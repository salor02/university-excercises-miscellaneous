from urllib import request
import nltk
from nltk.corpus import stopwords, wordnet as wn, wordnet_ic
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

text = 'mouse tea leave planet farm cat computer tobacco electricity'

#compara significati di base word con tutti i significati di compare word (non mi piace molto, poco efficiente)
def confidence_calc(base_word, compare_word):
    for meaning in base_word.keys():
        max = 0
        #trova il significato di compare word più simile e aggiunge questo valore alla confidence
        for compare_meaning in compare_word.keys():
            similarity = meaning.res_similarity(compare_meaning, brown_ic)
            if similarity > max:
                max = similarity
        base_word[meaning] = base_word[meaning] + max

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
brown_ic = wordnet_ic.ic('ic-brown.dat')

#dict delle parole
words = {}
for word in result:

    confidence = {}

    for meaning in wn.synsets(word, wn.NOUN):
        confidence[meaning] = 0

    words[word] = confidence

for word in words.values():
    for compare_word in words.values():
        if not compare_word == word:
            confidence_calc(word, compare_word)

#stampa
for item in words.keys():
    res = max(words[item], key=words[item].get)
    print(item + '-->' + res.definition())

