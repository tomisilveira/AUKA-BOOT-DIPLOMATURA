import gradio as gr
from pdf_processing import cargar_pdf, dividir_en_parrafos
from embeddings import generar_embeddings, responder_pregunta
from chatbot import generar_respuesta_natural

# Cargar el PDF y procesar el texto
ruta_pdf = "data/informacion_agencia.pdf"
texto_agencia = cargar_pdf(ruta_pdf)

if texto_agencia:
    parrafos = dividir_en_parrafos(texto_agencia)
    embeddings_parrafos, modelo_embeddings = generar_embeddings(parrafos)
else:
    raise ValueError("No se pudo cargar el PDF.")

# Función para manejar la interacción con el chatbot
def responder(pregunta, historial):
    if pregunta.strip() == "":
        return historial, "Por favor, escribe una pregunta."
    
    if pregunta.lower() in ['salir', 'adios', 'chau']:
        return historial + [{"role": "user", "content": pregunta}, {"role": "assistant", "content": "¡Hasta luego! Espero haberte sido de ayuda."}], ""
    
    # Mostrar "Respondiendo..." mientras se genera la respuesta
    yield historial + [{"role": "user", "content": pregunta}, {"role": "assistant", "content": "Respondiendo..."}], ""
    
    respuestas_relevantes = responder_pregunta(pregunta, parrafos, embeddings_parrafos, modelo_embeddings)
    
    if respuestas_relevantes:
        contexto = " ".join(respuestas_relevantes)
        respuesta = generar_respuesta_natural(pregunta, contexto)
        yield historial + [{"role": "user", "content": pregunta}, {"role": "assistant", "content": respuesta}], ""
    else:
        yield historial + [{"role": "user", "content": pregunta}, {"role": "assistant", "content": "No se encontró información relevante para tu pregunta."}], ""

# Tema personalizado con los colores proporcionados
tema_personalizado = gr.themes.Default(
    primary_hue="green",  # Color primario (#87B867)
    secondary_hue="blue",  # Color secundario (#2B3E4C)
    neutral_hue="gray",  # Color de fondo (#FCF2DC)
).set(
    body_background_fill="#FCF2DC",  # Fondo de la aplicación
    button_primary_background_fill="#87B867",  # Fondo del botón primario
    button_primary_text_color="#FFFFFF",  # Texto del botón primario (blanco)
    button_secondary_background_fill="#2B3E4C",  # Fondo del botón secundario
    button_secondary_text_color="#FFFFFF",  # Texto del botón secundario (blanco)
)

# CSS personalizado para ocultar los botones inferiores
css = """
footer {display: none !important;}
.gr-button {display: none !important;}
"""

# Interfaz de Gradio con el tema personalizado
with gr.Blocks(theme=tema_personalizado, css=css) as demo:
    gr.HTML("<script>document.title = 'Auka';</script>")
    # Título y descripción
    gr.Markdown("# Auka: Chatbot de la Agencia Neuquina de Innovación para el Desarrollo")
    gr.Markdown("¡Hola! Soy Auka, la mascota de la Agencia Neuquina de Innovación para el Desarrollo. ¿En qué puedo ayudarte hoy?")
    
    
    # Ventana de chat (parte superior)
    with gr.Row():
        historial = gr.Chatbot(label="Conversación", height=400, type="messages")  # Especificar type="messages"
    
    # Campo de texto (parte inferior)
    with gr.Row():
        pregunta = gr.Textbox(label="Escribe tu pregunta aquí", placeholder="¿Qué servicios ofrece la agencia?", lines=2)
    
    # Botones (debajo del campo de texto)
    with gr.Row():
        enviar = gr.Button("Enviar", variant="primary")  # Botón principal
        limpiar = gr.Button("Limpiar historial", variant="secondary")  # Botón secundario
    
    # Manejar el evento de enviar
    enviar.click(responder, inputs=[pregunta, historial], outputs=[historial, pregunta])
    
    # Manejar el evento de limpiar
    def limpiar_historial():
        return []
    limpiar.click(limpiar_historial, inputs=[], outputs=[historial])

    # JavaScript para cambiar el título de la pestaña
    demo.load(
        None,
        None,
        js="""
        document.title = "Auka";
        """
    )

# Lanzar la interfaz con el ícono personalizado
demo.launch(favicon_path="Auka.png")  # Cambia "Auka.png" por la ruta de tu archivo PNG