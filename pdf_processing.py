import pdfplumber

def cargar_pdf(ruta_pdf):
    """
    Extrae el texto de un archivo PDF usando pdfplumber.

    Args:
        ruta_pdf (str): Ruta del archivo PDF.

    Returns:
        str: Texto extraído del PDF.
    """
    try:
        texto = ""
        with pdfplumber.open(ruta_pdf) as pdf:
            for pagina in pdf.pages:
                texto += pagina.extract_text()
        
        if not texto.strip():
            raise ValueError("El PDF no contiene texto extraíble.")
        
        return texto
    except FileNotFoundError:
        return "El archivo PDF no se encontró."
    except Exception as e:
        return f"Error al cargar el PDF: {e}"


def dividir_en_parrafos(texto):
    """
    Divide el texto en párrafos, asegurando que no sean demasiado largos.

    Args:
        texto (str): Texto completo extraído del PDF.

    Returns:
        list: Lista de párrafos.
    """
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    parrafos = [p[:500] for p in parrafos]  # Limitar a 500 caracteres por párrafo
    return parrafos