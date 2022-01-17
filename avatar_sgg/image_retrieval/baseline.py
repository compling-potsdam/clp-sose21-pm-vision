from avatar_sgg.dataset.util import get_ade20k_split, get_categories
from avatar_sgg.image_retrieval.evaluation import compute_recall_on_category, calculate_normalized_cosine_similarity
from avatar_sgg.sentence_embedding.util import vectorize_captions
from avatar_sgg.sentence_embedding.distilbert_vectorizer import Vectorizer
from sentence_transformers import SentenceTransformer


def compute_average_similarity(ade20k_split, threshold=None):
    #vectorizer = Vectorizer()
    vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    stacked_vectors = vectorize_captions(ade20k_split, vectorizer)
    category = get_categories(ade20k_split)

    similarity = calculate_normalized_cosine_similarity(stacked_vectors)
    recall_val, mean_rank = compute_recall_on_category(similarity, category, threshold)

    for k in recall_val.keys():
        print(f"Recall @ {k}: {recall_val[k]}")
    print(f"Mean Rank{mean_rank}")


    return similarity.diag().mean()


if __name__ == "__main__":
    print("Start")
    _, _, test = get_ade20k_split()
    average_distance = compute_average_similarity(test, 0.7)
    print("average distance", average_distance)
    print("Done")
