import pandas as pd
import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_metadata = pd.read_csv("Proyecto peliculas/data/movies_metadata.csv")
df_metadataF = df_metadata[['belongs_to_collection', 'budget', 'genres', 'id', 'overview', 'popularity', 'poster_path', 'production_companies', 'release_date', 'revenue', 'runtime', 'tagline', 'title', 'vote_average', 'vote_count']]

df_keywords = pd.read_csv("Proyecto peliculas/data/keywords.csv")

df_cast = pd.read_csv("Proyecto peliculas/data/credits.csv")
df_castF = df_cast[['id',  'crew']]

df_rating = pd.read_csv("Proyecto peliculas/data/ratings.csv")

df = df_metadataF.dropna(subset=['title'])

print(df.describe())

df.id.astype('Int64')
df_cast.id.astype('Int64')

df['id']=df['id'].astype('Int64')
df_cast['id']=df_cast['id'].astype('Int64')


# Eliminar elementos repetidos en la columna 'id'
df_cast.drop_duplicates(subset=['id'], keep='first', inplace=True)
df.drop_duplicates(subset=['id'], keep='first', inplace=True)

dfP = pd.merge(df, df_cast, on='id', how="left", validate="one_to_one")

dfP = dfP.dropna(subset=['overview', 'cast'])



dfP['Tokens'] = dfP['title'] + dfP['overview'] + dfP['cast']

#print(dfP['cast'])

#dfP['cast'] = dfP['cast'].apply(lambda x: x.replace("None", "''"))
#dfP['cast'] = dfP['cast'].apply(lambda x: x.replace("\'", "\""))
#dfP['cast'] = dfP['cast'].apply(lambda x: json.loads(x))

#dfP['cast'] = dfP['cast'].apply(lambda x: [actor['name'] for actor in x])
#dfP['cast'] = dfP['cast'].apply(lambda x: ','.join(x))

#print(dfP['cast'])

from ast import literal_eval

features = ['cast', 'genres']
for feature in features:
    dfP[feature] = dfP[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        return i['name']
    



def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
    #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 10:
            names = names[:10]
    else:
        names = ""
    return names

dfP["Nombre"] = dfP['cast'].apply(get_list)
dfP['Nombre'] = dfP['Nombre'].apply(lambda x: ','.join(x))

dfP["Generos"] = dfP['genres'].apply(get_list)
dfP['Generos'] = dfP['Generos'].apply(lambda x: ', '.join(x))



dfP['Tokens'] = "Title: " +  dfP['title'] + '. Overview: ' + dfP['overview'] + '. Genres: ' + dfP['Generos']

print("Tokens:")
print(dfP["Tokens"][27])

dfP['Tokens'][:3000].to_csv('movies.csv', index=False) 
dfP.to_csv('moviesFull.csv', index=False)

#Empezamos a tokenizar

def preparar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\W+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

dfP["Tokens"] = dfP['Tokens'].apply(preparar_texto)

# Tokenización: Divide el texto en oraciones y luego en palabras
def tokenizar(texto):
   
    if not texto:    
        texto = ''
    
    tokens = word_tokenize(texto)
    stopwords_sp = set(stopwords.words('english'))
    tokens_limpios = [token for token in tokens if token.lower() not in stopwords_sp]

    return tokens_limpios

dfP['Tokens'] = dfP['Tokens'].apply(lambda x:tokenizar(x))



def stemizar(tokens):
    stemmer = SnowballStemmer('english')

    tokensL = [stemmer.stem(x) for x in tokens]

    return tokensL

dfP['TokensL'] = dfP['Tokens'].apply(stemizar)


def procesar_tfidf(documentos):
    vectorizador = TfidfVectorizer()
    tfidf_matrix = vectorizador.fit_transform(documentos)
    return tfidf_matrix, vectorizador

def busqueda(tfidf_matrix, vectorizador, documento):
    doc_vectorizado = vectorizador.transform(documento)
    similitudes = cosine_similarity(doc_vectorizado, tfidf_matrix)

    # Obtener inidices de máximo indice de similaridad (excluyendo la película seleccionada)
    similar_movie_indices = similitudes.argsort()[0][::-1][1:]

    # Obtener top N similar movies
    top_N = 10  # Numero desdea de recomendaciones
    recommended_movies = df.iloc[similar_movie_indices[:top_N]]
    print(recommended_movies.columns)
    return recommended_movies['title']

tfidf_matrix, vectorizador = procesar_tfidf(dfP['TokensL'].apply(lambda x: ' '.join(x)))

print("Busqueda:")
print(busqueda(tfidf_matrix, vectorizador, dfP['Tokens'][1]))

print("tokens de 17:")
print(dfP['Tokens'][1])