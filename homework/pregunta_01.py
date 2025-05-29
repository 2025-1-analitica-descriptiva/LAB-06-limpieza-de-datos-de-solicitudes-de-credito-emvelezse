"""
Escriba el codigo que ejecute la accion solicitada en la pregunta.
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import os
import pandas as pd


def create_normalized_key(df, column_name):
    """Cree una nueva columna en el DataFrame que contenga
    el key de la columna 'raw_text'"""

    df[column_name] = df[column_name].str.lower()

    df[column_name] = df[column_name].str.replace(r"[_-]", " ", regex=True)
    df = df.copy()

    # Copie la columna 'text' a la columna 'key'
    df["key"] = df[column_name]

    # Remueva los espacios en blanco al principio y al final de la cadena
    df["key"] = df["key"].str.strip()

    # Convierta el texto a minúsculas
    df["key"] = df["key"].str.lower()

    # Transforme palabras que pueden (o no) contener guiones por su
    # version sin guion (este paso es redundante por la linea siguiente.
    # Pero es claro anotar la existencia de palabras con y sin '-'.
    # Reemplazarlos por espacios


    # Remueva puntuación y caracteres de control
    df["key"] = df["key"].str.translate(
        str.maketrans("", "", "!\"#$%&'()*+,¿./:;<=>?@[\\]^`{|}~")
    )

    # Convierta el texto a una lista de tokens
    df["key"] = df["key"].str.split()    

    # Transforme cada palabra con un stemmer de Porter
    stemmer = nltk.PorterStemmer()
    df["key"] = df["key"].apply(lambda x: [stemmer.stem(word) for word in x])

    # Ordene la lista de tokens y remueve duplicados
    df["key"] = df["key"].apply(lambda x: sorted(set(x)))

    # Convierta la lista de tokens a una cadena de texto separada por espacios
    df["key"] = df["key"].str.join(" ")

    return df

def generate_cleaned_text(df, column_name):
    """Crea la columna 'cleaned_text' en el DataFrame"""

    keys = df.copy()

    # Ordene el dataframe por 'key' y 'text'
    keys = keys.sort_values(by=["key", column_name], ascending=[True, True])

    # Seleccione la primera fila de cada grupo de 'key'
    keys = df.drop_duplicates(subset="key", keep="first")

    # Cree un diccionario con 'key' como clave y 'text' como valor
    key_dict = dict(zip(keys["key"], keys[column_name]))

    # Cree la columna 'cleaned' usando el diccionario
    df["cleaned_text"] = df["key"].map(key_dict)

    df = df.drop(columns =["key", column_name])
    # df = df.drop(columns =["key"])

    df = df.rename(columns={"cleaned_text": column_name})

    return df


def eliminar_stopwords(texto):
    stop_words = set(stopwords.words('spanish'))
    palabras = word_tokenize(texto.lower())
    palabras_filtradas = [palabra for palabra in palabras if palabra.isalpha() and palabra not in stop_words]
    return ' '.join(palabras_filtradas)

def normalizar_fecha_o_none(fecha):
    if isinstance(fecha, str) and '/' in fecha:
        partes = fecha.split('/')
        if len(partes) == 3 and len(partes[0]) == 4:
            anio, mes, dia = partes
            if int(mes)==int(dia):
                return f"{dia.zfill(2)}/{mes.zfill(2)}/{anio}"
            elif int(mes) <= 12 and int(dia) <= 12:
                # Fecha ambigua, descartamos
                
                return f"{dia.zfill(2)}/{mes.zfill(2)}/{anio}"
            elif int(mes) > 12 and int(dia) > 12:
                return None
            else:
                # Fecha no ambigua, normalizamos
                return f"{dia.zfill(2)}/{mes.zfill(2)}/{anio}"
                # return None
        else:
            return fecha
        
def save_output(name:str, dataframe:pd.DataFrame):
    output_directory = "files/output"
    if not os.path.exists(output_directory):    
        os.makedirs(output_directory)
    dataframe.to_csv(f"{output_directory}/{name}", sep=";")

def pregunta_01():
    """
    Realice la limpieza del archivo "files/input/solicitudes_de_credito.csv".
    El archivo tiene problemas como registros duplicados y datos faltantes.
    Tenga en cuenta todas las verificaciones discutidas en clase para
    realizar la limpieza de los datos.

    El archivo limpio debe escribirse en "files/output/solicitudes_de_credito.csv"

    """
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    df = pd.read_csv("files/input/solicitudes_de_credito.csv", sep=";")
    df = df.drop(columns=['Unnamed: 0'])
    df = df.dropna()
    df["comuna_ciudadano"] = df["comuna_ciudadano"].apply(lambda x: int(x))
    df["sexo"] = df["sexo"].str.lower()
    df["tipo_de_emprendimiento"] = df["tipo_de_emprendimiento"].str.lower()
    df = create_normalized_key(df, "idea_negocio")
    df = generate_cleaned_text(df, "idea_negocio")
    df["idea_negocio"] = df["idea_negocio"].apply(eliminar_stopwords)
    df["barrio"] = df["barrio"].str.lower()
    df["barrio"] = df["barrio"].str.replace(r"[_-]", " ", regex=True)
    df["monto_del_credito"] = df["monto_del_credito"].map(
    lambda x: x.replace("$", "").strip() if isinstance(x, str) else x
    )
    df["monto_del_credito"] = df["monto_del_credito"].apply(
    lambda x: float(x.replace(",", "")) if isinstance(x, str) else x
    )
    df = create_normalized_key(df, "línea_credito")
    df = generate_cleaned_text(df, "línea_credito")
    df["línea_credito"] = df["línea_credito"].apply(eliminar_stopwords)
    df['fecha_de_beneficio'] = df['fecha_de_beneficio'].apply(normalizar_fecha_o_none)
    df = df[df['fecha_de_beneficio'].notna()]
    df['fecha_de_beneficio'] = pd.to_datetime(df['fecha_de_beneficio'], format='%d/%m/%Y', errors='coerce')
    df = df[df['fecha_de_beneficio'].notna()]
    df = df.drop_duplicates()

    save_output("solicitudes_de_credito.csv", df)


    