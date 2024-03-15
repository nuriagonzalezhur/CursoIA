import pandas as pd
import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import requests
from astrapy.db import AstraDB
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import openai

load_dotenv()

openai_secret = os.getenv('OPENAI_API_KEY')
datastax_url = os.getenv('URL_DATASTAX')
datastax_secret = os.getenv('DATASTAX')

openai.api_key = openai_secret

print(openai_secret)

st.title('MovIA')

embeddings = OpenAIEmbeddings(api_key=openai_secret, model="text-embedding-3-large")

ruta = os.getcwd()
print(f"Ruta", {ruta})

#Empezamos a crear streamlit
@st.cache_data
def load_titulos():
    return pd.read_csv('c:/Users/Admin/Documents/Curso IA/Repositorio/CursoIA/Proyecto peliculas/moviesFull.csv')['title']
    
movies_titulos = load_titulos()

print(movies_titulos)

db = AstraDB(
  token=datastax_secret,
  api_endpoint=datastax_url)
                              
print(f"Connected to Astra DB: {db.get_collections()}")
collection = db.collection("vector_movies")
print(collection)

def get_embedding(text):
    query_result = embeddings.embed_query(text)
    return query_result

def buscar(input):
    embedding_busqueda = get_embedding(input)
    results = collection.vector_find(embedding_busqueda, limit=3, fields={"text", "$vector"})
    st.write(type(results))
    if len(results) > 1:
        return results[1:]
    else:
        # Retorna una lista vacía o maneja la situación como prefieras
        return ['No hay recomendaciones']
    

url_omdb = "http://www.omdbapi.com/?apikey=" + os.getenv("OMDB_API_KEY") + "&t="

def get_poster(movie):
    """ 
    Devuelve la ruta al poster en jpg
    
    Args:
        imdbId (str): El identificador IBMDB de la película
    """
    result = requests.get(url_omdb + movie)
    result_json = result.json()
    return result_json["Poster"]

speech_file_path = 'speech.mp3'



def process_movie_speech(movie):
    #docs = db.similarity_search(movie)

    response = openai.audio.speech.create(
    model="tts-1",
    voice="onyx",
    input=f"Sinopsis: {movie}"
    )
    response.write_to_file(speech_file_path)
    return speech_file_path

def describirPelicula(title):

    response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Dame una descripción de la siguiente película: {title}"},
                            
                        ]
                    }
                ],
                max_tokens=300,
            )
    speech_file_path = process_movie_speech(response.choices[0].message.content)

    # Mostrar el archivo de audio en Streamlit
    st.audio(str(speech_file_path), format='audio/mp3')

opcion_seleccionada = st.selectbox(
    "¿Qué película te ha gustado?",
    movies_titulos,
    index=None,
    placeholder="Selecciona una película...",
)

if opcion_seleccionada:
    # Encontrar el id de la película seleccionada usando el diccionario
    with st.status("Buscando recomendaciones...") as s:
       resultado = buscar(opcion_seleccionada)
       st.write("Buscando...")
       s.update(label="Encontrado!")
    
        
    for sugerencia in resultado:
    
        title = sugerencia['text'].split(". Overview:")[0].replace("Title: ", "")
        overview = sugerencia['text'].split(". Overview: ")[1].split(". Genres: ")[0]
        
        peli = sugerencia['text']
        
        st.subheader(title, divider='rainbow')
        if ". Genres: " in sugerencia['text']:
            genres = sugerencia['text'].split(". Genres: ")[1]
        else:
            genres = "No hay géneros"  # O cualquier otro valor predeterminado o manejo de error que prefieras

        st.markdown(f"Géneros &mdash;\
            {genres} :movie_camera:")
        st.divider()  

        st.write(overview)

        st.divider()
        st.image(get_poster(title), caption=title)

        describirPelicula(title)



    