import torch, collections


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


def compute_recall_johnson_feiefei(similarity, threshold=None, recall_at: list = [1, 2, 5, 8, 10, 20]):
    """
    This is how I understood recall computation from  https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf, p.6
    For each image, we know what is the expected best result (gold image) for a given text query.
    If we consider the best k results as in @K metrics, we have to check if our image is included in the top k.
    We can then compute a proportion of times our system includes the true image and also calculate the mean rank recomandation
    of the gold image.

    :param similarity:
    :param threshold:
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

    recall_val = {k: ((gold_ranks < k).sum().type(torch.float) / number_entries) for k in recall_at if
                  k <= number_entries}

    return recall_val, mean_rank


def compute_recall_on_category(similarity, category, threshold=None, recall_at: list = [1, 2, 5, 8, 10, 20]):
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

    recall_val = {k: (numerator(k) / denominator(k)) for k in recall_at if k <= number_entries}

    return recall_val, mean_rank
