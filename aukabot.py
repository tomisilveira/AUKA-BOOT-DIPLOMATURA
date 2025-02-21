import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Cargar la API key desde el archivo .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 1. Cargar y procesar el PDF
def cargar_pdf(ruta_pdf):
    """
    Extrae el texto de un archivo PDF.
    """
    try:
        with open(ruta_pdf, "rb") as archivo:
            lector = PyPDF2.PdfReader(archivo)
            if len(lector.pages) == 0:
                raise ValueError("El PDF no contiene páginas.")
            texto = ""
            for pagina in lector.pages:
                texto += pagina.extract_text()
        return texto
    except FileNotFoundError:
        st.error("El archivo PDF no se encontró.")
    except Exception as e:
        st.error(f"Error al cargar el PDF: {e}")
    return ""


# 2. Dividir el texto en párrafos
def dividir_en_parrafos(texto):
    """
    Divide el texto en párrafos, asegurando que los párrafos no sean demasiado largos.
    """
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    # Limitar la longitud de cada párrafo para mejorar la eficiencia
    parrafos = [p[:500] for p in parrafos]  # Limitar a 500 caracteres por párrafo
    return parrafos

# 3. Generar embeddings para los párrafos
def generar_embeddings(parrafos):
    """
    Convierte los párrafos en embeddings usando un modelo de Sentence Transformers.
    """
    try:
        modelo_embeddings = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = modelo_embeddings.encode(parrafos)
        return embeddings, modelo_embeddings
    except Exception as e:
        st.error(f"Error al generar embeddings: {e}")
        return None, None

# 4. Responder preguntas basadas en el PDF
def responder_pregunta(pregunta, parrafos, embeddings_parrafos, modelo_embeddings, n_resultados=3):
    """
    Encuentra los párrafos más relevantes para una pregunta dada.
    """
    try:
        # Convertir la pregunta en un embedding
        embedding_pregunta = modelo_embeddings.encode([pregunta])
        
        # Calcular la similitud entre la pregunta y los párrafos
        similitudes = cosine_similarity(embedding_pregunta, embeddings_parrafos).flatten()
        
        # Obtener los índices de los párrafos más relevantes
        indices_relevantes = similitudes.argsort()[-n_resultados:][::-1]
        
        # Devolver los párrafos más relevantes
        return [parrafos[i] for i in indices_relevantes]
    except Exception as e:
        st.error(f"Error al responder la pregunta: {e}")
        return []

# 5. Generar una respuesta natural usando la API de Gemini
def generar_respuesta_natural(pregunta, contexto):
    """
    Usa la API de Gemini para generar una respuesta natural basada en el contexto.
    """
    try:
        # Limitar la longitud del contexto si es necesario
        if len(contexto) > 2000:
            contexto = contexto[:2000]  # Limitar el contexto a 2000 caracteres
        
        prompt = f"Pregunta: {pregunta}\nContexto: {contexto}\nRespuesta:"
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        st.error(f"Error al generar la respuesta: {e}")
        return "Lo siento, no pude generar una respuesta en este momento."


# 6. Interfaz gráfica con Streamlit
def main():
    """
    Interfaz gráfica del chatbot usando Streamlit.
    """
    # Configuración de la página
    st.set_page_config(
        page_title="Auka: Chatbot de la Agencia Neuquina de Innovación para el Desarrollo",
        page_icon="🤖",
        layout="centered",
    )
    
    # Título y descripción
    st.title("Auka: Chatbot de la Agencia Neuquina de Innovación para el Desarrollo")
    st.write("¡Hola! Soy Auka, la mascota de la Agencia Neuquina de Innovación para el Desarrollo. ¿En qué puedo ayudarte hoy?")
    
    # Inicializar el historial de conversaciones en la sesión
    if "historial" not in st.session_state:
        st.session_state.historial = []
    
    # Cargar el PDF y procesar el texto
    ruta_pdf = "data/informacion_agencia.pdf"
    texto_agencia = cargar_pdf(ruta_pdf)
    
    if texto_agencia:
        parrafos = dividir_en_parrafos(texto_agencia)
        embeddings_parrafos, modelo_embeddings = generar_embeddings(parrafos)
        
        if embeddings_parrafos is not None:
            # Campo de entrada para la pregunta del usuario
            pregunta_usuario = st.text_input("Tú: ", "")
            
            if pregunta_usuario:
                if pregunta_usuario.lower() in ['salir', 'adios', 'chau']:
                    st.write("Auka: ¡Hasta luego! Espero haberte sido de ayuda.")
                else:
                    # Obtener los párrafos más relevantes para la pregunta
                    respuestas_relevantes = responder_pregunta(pregunta_usuario, parrafos, embeddings_parrafos, modelo_embeddings)
                    
                    if respuestas_relevantes:
                        # Combinar los párrafos relevantes en un solo contexto
                        contexto = " ".join(respuestas_relevantes)
                        
                        # Generar una respuesta natural basada en el contexto
                        respuesta = generar_respuesta_natural(pregunta_usuario, contexto)
                        
                        # Guardar la pregunta y respuesta en el historial
                        st.session_state.historial.append({"pregunta": pregunta_usuario, "respuesta": respuesta})
                        
                        # Mostrar la respuesta
                        st.write(f"Auka: {respuesta}")
                    else:
                        st.warning("No se encontró información relevante para tu pregunta.")
    
    # Mostrar el historial de conversaciones
    if st.session_state.historial:
        st.subheader("Historial de Conversaciones")
        for conversacion in st.session_state.historial:
            st.write(f"**Tú:** {conversacion['pregunta']}")
            st.write(f"**Auka:** {conversacion['respuesta']}")
            st.write("---")
    # Limpiar historial
    if st.button("Limpiar historial"):
        st.session_state.historial = []
        st.write("Historial de conversaciones limpiado.")

    # Simulación de respuesta
    if pregunta_usuario:
        with st.spinner("Auka está pensando..."):
            respuesta = generar_respuesta_natural(pregunta_usuario, contexto)
        st.write(f"Auka: {respuesta}")

# Ejecutar la aplicación de Streamlit
if __name__ == "__main__":
    main()