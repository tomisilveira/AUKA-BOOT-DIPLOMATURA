from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def generar_embeddings(parrafos):
    """
    Genera embeddings para los párrafos usando el modelo all-MiniLM-L6-v2.

    Args:
        parrafos (list): Lista de párrafos.

    Returns:
        tuple: (embeddings, modelo_embeddings)
    """
    try:
        modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = modelo_embeddings.encode(parrafos)
        return embeddings, modelo_embeddings
    except Exception as e:
        return None, None


def responder_pregunta(pregunta, parrafos, embeddings_parrafos, modelo_embeddings, n_resultados=3):
    """
    Encuentra los párrafos más relevantes para una pregunta dada.

    Args:
        pregunta (str): Pregunta del usuario.
        parrafos (list): Lista de párrafos.
        embeddings_parrafos (array): Embeddings de los párrafos.
        modelo_embeddings (SentenceTransformer): Modelo de embeddings.
        n_resultados (int): Número de párrafos relevantes a devolver.

    Returns:
        list: Párrafos más relevantes.
    """
    try:
        embedding_pregunta = modelo_embeddings.encode([pregunta])
        similitudes = cosine_similarity(embedding_pregunta, embeddings_parrafos).flatten()
        indices_relevantes = similitudes.argsort()[-n_resultados:][::-1]
        return [parrafos[i] for i in indices_relevantes]
    except Exception as e:
        return []