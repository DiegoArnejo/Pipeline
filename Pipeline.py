import nltk
import string
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

'''
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')
'''

ingles = set(stopwords.words("english")) 

def get_wordnet_pos(tag):
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag[0].upper(), wordnet.NOUN)

def quitarStopwords_eng(texto):
    excepciones = {"Python", "JavaScript", "CPlus", "Rust", "Java", "Go"}
    #Excluyo los nombres de los lenguajes para que no pasen a minuscula, es en vano lematizarlos
    
    texto_limpio = [w if w in excepciones else w.lower() 
                    for w in texto if w.lower() not in ingles 
                    and w not in string.punctuation 
                    and w not in [".-"]]
    return texto_limpio

def lematizar(texto):
   pos_tags = nltk.pos_tag(texto)  # Etiqueto todo el texto 
   texto_lema = [lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags]
   return texto_lema

#Inicializo el Lematizador
lemmatizer = WordNetLemmatizer()

corpus = [
lematizar(quitarStopwords_eng(word_tokenize("Python is an interpreted and high-level language, while CPlus is a compiled and low-level language .-"))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security .-"))),
lematizar(quitarStopwords_eng(word_tokenize("Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript is widely used in web development, while Go is ideal for servers and cloud applications."))),
lematizar(quitarStopwords_eng(word_tokenize("Python is slower than CPlus and Rust due to its interpreted nature."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science .-"))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript does not require compilation, while CPlus and Rust require code compilation before execution .-"))),
lematizar(quitarStopwords_eng(word_tokenize("Python and JavaScript have large communities and an extensive number of available libraries."))),
lematizar(quitarStopwords_eng(word_tokenize("Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers.")))
]

tokens_totales = []

for oracion in corpus:
    tokens_totales.extend(oracion)

frecuencia = FreqDist(tokens_totales)

corpus_texto = [" ".join(oracion) for oracion in corpus]

print("\nCorpus procesado:\n")
for doc in corpus:
    print(doc)
print("-"*75)

# Tf-idf Vectorizacion
vectorizer = TfidfVectorizer() 
X = vectorizer.fit_transform(corpus_texto)
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print("Matriz TF-IDF:\n")
print(df)
print("\nPalabras utilizadas por el modelo TF-IDF:\n")
print(vectorizer.get_feature_names_out())
print("-"*75)

#Lematizadas
print("Palabras lematizadas y su frecuencia:\n")
for palabra, freq in frecuencia.most_common():
    print(f"{palabra:<15} {freq:>5}")

#Grafico
frecuencia.plot(20,show=True)
