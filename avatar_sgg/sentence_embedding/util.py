import torch

def vectorize_captions(ade20k_split, vectorizer, caption_key = "caption"):
    """

    :param ade20k_split:
    :param vectorizer: As in Vectorizer in distilbert_vectorizer.py or a SentenceTransformer model used in SentenceBert
    :return:
    """
    stacked_vectors = None
    for image in ade20k_split:
        vectors = vectorizer.encode(ade20k_split[image][caption_key], convert_to_tensor= True)
        # adds a dimension at position 0; dimension 0 is used to "list the entries"
        if len(vectors.shape) < 3:
            vectors = vectors.unsqueeze(0)
        if stacked_vectors is None:
            stacked_vectors = vectors
        else:
            stacked_vectors = torch.cat((stacked_vectors, vectors), dim=0)

    return stacked_vectors