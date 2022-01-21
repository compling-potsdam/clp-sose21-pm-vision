import torch, collections
from sentence_transformers import SentenceTransformer
from avatar_sgg.sentence_embedding.util import vectorize_captions
from avatar_sgg.dataset.util import get_categories
from avatar_sgg.captioning.catr.inference import CATRInference
import string

def calculate_normalized_cosine_similarity(input):
    """
    Input has the dimensions: number of entries * 2 * vector dimension
    :param input:
    :return:
    """

    first_embeds = input[:, 0, :]
    second_embeds = input[:, 1, :]

    similarity = first_embeds @ second_embeds.T

    norm_1 = first_embeds.norm(dim=1, p=2)
    norm_2 = second_embeds.norm(dim=1, p=2)
    norm = norm_1.unsqueeze(1) * norm_2.unsqueeze(1).T
    # Normalize the similarity scores to make them comparable
    similarity = similarity / norm

    return similarity


def compute_recall_johnson_feiefei(similarity, threshold, category,  recall_at: list = [1, 2, 3, 4, 5,  10, 20, 50, 100]):
    """
        This is how I understood recall computation from  https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf, p.6
        For each image, we know what is the expected best result (gold image) for a given text query.
        If we consider the best k results as in @K metrics, we have to check if our image is included in the top k.
        We can then compute a proportion of times our system includes the true image and also calculate the mean rank recomandation
        of the gold image.

    :param similarity:
    :param threshold:
    :param category: Not used. Just to keep the same signature with compute_recall_on_category
    :param recall_at:
    :return:
    """
    number_entries = similarity.shape[0]
    values, ranks = torch.topk(similarity, number_entries)
    gold_recommendations = torch.arange(0, number_entries, dtype=ranks.dtype, device=ranks.device).unsqueeze(1)

    # dimension 0 is the entry dimension, dimension 1 is the ranking for a given entry
    entry_ranks, gold_ranks = (ranks == gold_recommendations).nonzero(as_tuple=True)
    mean_rank = (gold_ranks + 1).type(torch.float).mean()

    if threshold:
        threshold_mask = (values >= threshold)
        # Due to the threshold, you might have less entries returned than number_entries
        entry_ranks, gold_ranks = torch.logical_and((ranks == gold_recommendations), threshold_mask).nonzero(
            as_tuple=True)

    recall_val = {"recall_at"+str(k): ((gold_ranks < k).sum().type(torch.float) / number_entries).to("cpu").numpy() for k in recall_at if
                  k <= number_entries}

    mean_rank = mean_rank.to("cpu").numpy()

    return recall_val, mean_rank


def compute_recall_on_category(similarity, threshold, category,  recall_at: list = [1, 2, 3, 4, 5,  10, 20, 50, 100]):
    """
        For each image, we know what is the expected best result (gold image) for a given text query.
        That gives us a gold category annotation for the current image. We can find out how many images with this particular
        category are in the current comparison. The recall is the proportion of the number of images from this particular
        category in the top k recomendation divided by k. If k is superior to the sum of all images of this given category
        in the current comparison, k is replaced by this sum. For example, if we are looking for an image of a bathroom,
        if we have only 3 bathrooms out of 10 images, and if we are able to retrieve all bathrooms in a recall @k = 5,
        we compute 3 bathrooms out of 3 bathrooms instead of 3 bathrooms out of 5 recommendations.
    :param similarity:
    :param threshold:
    :param category:
    :param recall_at:
    :return:
    """

    number_entries = similarity.shape[0]

    assert len(category) == number_entries

    category_to_entry_lookup = collections.defaultdict(list)
    for k, v in category.items():
        category_to_entry_lookup[v].append(k)

    category_to_idx = {c: i for i, c in enumerate(sorted(category_to_entry_lookup.keys()))}
    vectorized_category = [category_to_idx[category[k]] for k in sorted(category.keys())]

    values, ranks = torch.topk(similarity, number_entries)
    gold_recommendations = torch.arange(0, number_entries, dtype=ranks.dtype, device=ranks.device).unsqueeze(-1)
    gold_category_recommendations = torch.tensor(vectorized_category, dtype=ranks.dtype, device=ranks.device)
    category_ranks = torch.gather(gold_category_recommendations.repeat((number_entries, 1)), 1, ranks)
    # dimension 0 is the entry dimension, dimension 1 is the ranking for a given entry
    entry_ranks, gold_ranks = (ranks == gold_recommendations).nonzero(as_tuple=True)
    mean_rank = (gold_ranks + 1).type(torch.float).mean()

    # Unsqueeze here to make each value in gold_category_recommendations compared against all rows in category ranks
    gold_category_recommendations = gold_category_recommendations.unsqueeze(-1)
    category_mask = (gold_category_recommendations == category_ranks)
    category_mask_unchanged_by_threshold = (gold_category_recommendations == category_ranks)
    if threshold:
        threshold_mask = (values >= threshold)
        # Due to the threshold, you might have less entries returned than number_entries
        category_mask = torch.logical_and(category_mask, threshold_mask)

    target_dim = 1
    entries_tensor = torch.ones_like(gold_category_recommendations).type(torch.float).squeeze()
    sub_total_gold_category = category_mask_unchanged_by_threshold.sum(dim=target_dim).type(torch.float)
    # this lambda counts the number of gold category per entry in the top k
    numerator = lambda k: category_mask[:, :k].sum(dim=target_dim).sum().type(torch.float)

    # this lambda counts the number of possible gold category per entry. If the number of possible gold category is
    # superior to k, then the number is defaulted to k.
    denominator = lambda k: torch.where(sub_total_gold_category <= entries_tensor * k, sub_total_gold_category,
                                        entries_tensor * k).sum()
    recall_val = {"recall_at_"+str(k): (numerator(k) / denominator(k)).to("cpu").numpy() for k in recall_at if k <= number_entries}
    mean_rank = mean_rank.to("cpu").numpy()
    return recall_val, mean_rank

def compute_similarity(ade20k_split, threshold=None, recall_funct=compute_recall_johnson_feiefei):
    """

    :param ade20k_split:
    :param threshold:
    :param recall_funct:
    :return:
    """
    vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    stacked_vectors = vectorize_captions(ade20k_split, vectorizer)
    category = get_categories(ade20k_split)

    similarity = calculate_normalized_cosine_similarity(stacked_vectors)
    recall_val, mean_rank = recall_funct(similarity, threshold, category)

    for k in recall_val.keys():
        print(f"{k}: {recall_val[k]}")

    recall_val["mean_rank"] = mean_rank
    print(f"Mean Rank: {mean_rank}")

    average_similarity = similarity.diag().mean().to("cpu").numpy()

    print(f"Average Similarity: {average_similarity}")

    recall_val["average_similarity"] = average_similarity
    recall_val["threshold"] = threshold
    return recall_val

def compute_average_similarity(ade20k_split, threshold=None, recall_funct=compute_recall_johnson_feiefei):
    """
    Pre-requisite. The ade20k_split has been enriched with 'add_inferred_captions()'. The synthetic caption are used as
    query to retrieve the images based on the human captions. The results are averaged by the number of human captions
    available (2 only)
    :param ade20k_split:
    :param threshold:
    :param recall_funct:
    :return:
    """
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
    recall_val, mean_rank = recall_funct(similarity_caption_1, threshold, category)
    recall_val_2, mean_rank_2 = recall_funct(similarity_caption_2, threshold, category)

    print(f"Threshold for retrieval: {threshold}")
    for k in recall_val.keys():
        print(f"Average {k}: {(recall_val[k] + recall_val_2[k]) / 2}")

    average_mean_rank = ((mean_rank + mean_rank_2) / 2)
    recall_val["mean_rank"] = average_mean_rank

    print(f"Average Mean Rank: {average_mean_rank}")
    average_similarity = ((similarity_caption_1.diag().mean() + similarity_caption_2.diag().mean()) / 2).to("cpu").numpy()
    print(f"Average Similarity{average_similarity}")

    recall_val["average_similarity"] = average_similarity
    recall_val["threshold"] = threshold

    return recall_val


def merge_human_captions(data_split):
    """
    Merges all Human captions together, let the CATR caption separate.
    :param data_split:
    :return:
    """

    for path in data_split.keys():

        if len(data_split[path]["caption"]) == 3:
            human_captions = data_split[path]["caption"][:2]
            catr_caption = data_split[path]["caption"][2]
            glue = ". "
            if human_captions[0][-1] in string.punctuation:
                glue = " "
            human_captions = glue.join(human_captions)

            data_split[path]["caption"] = [ human_captions , catr_caption]


def add_inferred_captions(data_split):
    """
    Adds caption from CATR model to the given ADE20K split.
    :param data_split:
    :return:
    """
    catr = CATRInference()
    for path in data_split.keys():
        output = catr.infer(path)
        data_split[path]["caption"].append(output)

