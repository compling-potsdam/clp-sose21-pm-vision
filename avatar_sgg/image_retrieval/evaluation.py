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

    assert len(category) == number_entries

    category_to_entry_lookup = collections.defaultdict(list)
    for k, v in category.items():
        category_to_entry_lookup[v].append(k)

    idx_to_category = {i: c for i, c in enumerate(sorted(category_to_entry_lookup.keys()))}
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
    if threshold:
        threshold_mask = (values >= threshold)
        # Due to the threshold, you might have less entries returned than number_entries
        category_mask = torch.logical_and(category_mask, threshold_mask)

    target_dim = 1
    recall_val = {k: (category_mask[:, :k].sum(dim=target_dim).sum().type(torch.float) / (number_entries * k)) for k in
                  recall_at if
                  k <= number_entries}

    return recall_val, mean_rank


def compute_recall(similarity, threshold=None, recall_at: list = [1, 2, 5, 8, 10, 20]):
    if not threshold:
        # The mean of the diagonal is the criteria to define something "meaningful"
        pred_rank = (similarity >= similarity.diag().mean().view(-1, 1)).sum(-1)
    else:
        pred_rank = (similarity >= (torch.ones_like(similarity.diag()) * threshold).view(-1, 1)).sum(-1)
    num_sample = pred_rank.shape[0]
    for k in recall_at:
        if k <= num_sample:
            print('Recall @ %d: %.4f; ' % (k, float((pred_rank < k).sum()) / num_sample))
