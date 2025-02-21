import google.generativeai as genai
from dotenv import load_dotenv
import os

# Cargar la API key desde el archivo .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def generar_respuesta_natural(pregunta, contexto):
    """
    Genera una respuesta natural y detallada usando la API de Gemini.

    Args:
        pregunta (str): Pregunta del usuario.
        contexto (str): Contexto relevante para la pregunta.

    Returns:
        str: Respuesta generada.
    """
    try:
        if len(contexto) > 2000:
            contexto = contexto[:2000]  # Limitar el contexto a 2000 caracteres
        
        # Mejorar el prompt para obtener respuestas más detalladas
        prompt = (
            "Eres un asistente virtual de la Agencia Neuquina de Innovación para el Desarrollo. "
            "Tu tarea es responder preguntas de manera clara, detallada y amigable, utilizando el siguiente contexto:\n\n"
            f"Contexto: {contexto}\n\n"
            f"Pregunta: {pregunta}\n\n"
            "Por favor, responde de manera completa y explayada, asegurándote de incluir todos los detalles relevantes. "
            "Si es necesario, proporciona ejemplos o explicaciones adicionales para que la respuesta sea más útil."
        )
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Lo siento, no pude generar una respuesta en este momento."