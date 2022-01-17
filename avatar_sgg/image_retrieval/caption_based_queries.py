from avatar_sgg.dataset.util import get_ade20k_split, get_categories
from avatar_sgg.image_retrieval.evaluation import compute_recall_on_category, calculate_normalized_cosine_similarity
from avatar_sgg.sentence_embedding.util import vectorize_captions
from avatar_sgg.sentence_embedding.distilbert_vectorizer import Vectorizer
from sentence_transformers import SentenceTransformer
from avatar_sgg.captioning.catr.inference import CATRInference
import torch


def compute_average_similarity(ade20k_split, threshold=None):
    # vectorizer = Vectorizer()
    vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    stacked_vectors = vectorize_captions(ade20k_split, vectorizer)
    category = get_categories(ade20k_split)

    index_caption_1 = 0
    index_caption_2 = 1
    index_inferred_caption = 2
    caption_dim = 1

    comparison_1 = torch.cat((stacked_vectors[:, index_caption_1, :].unsqueeze(caption_dim),
                              stacked_vectors[:, index_inferred_caption, :].unsqueeze(caption_dim)), dim=caption_dim)

    comparison_2 = torch.cat((stacked_vectors[:, index_caption_2, :].unsqueeze(caption_dim),
                              stacked_vectors[:, index_inferred_caption, :].unsqueeze(caption_dim)), dim=caption_dim)

    similarity_caption_1 = calculate_normalized_cosine_similarity(comparison_1)
    similarity_caption_2 = calculate_normalized_cosine_similarity(comparison_2)
    recall_val, mean_rank = compute_recall_on_category(similarity_caption_1, category, threshold)
    recall_val_2, mean_rank_2 = compute_recall_on_category(similarity_caption_2, category, threshold)

    for k in recall_val.keys():
        print(f"Average Recall @ {k}: {(recall_val[k] + recall_val_2[k]) / 2}")
    print(f"Average Mean Rank{(mean_rank + mean_rank_2) / 2}")

    return (similarity_caption_1.diag().mean() + similarity_caption_2.diag().mean()) / 2


if __name__ == "__main__":

    catr = CATRInference()

    print("Start")
    train, dev, test = get_ade20k_split()
    current = dev

    for path in current.keys():
        output = catr.infer(path)
        current[path]["caption"].append(output)

    average_distance = compute_average_similarity(current)
    print("average distance", average_distance)
    print("Done")
