import torch

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
    #Normalize the similarity scores to make them comparable
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
    gold_recommendations = torch.arange(0, number_entries, dtype=torch.int64, device=similarity.device).unsqueeze(1)
    gold_ranks = (ranks == gold_recommendations).nonzero(as_tuple=True)

    # dimension 0 is the entry dimension, dimension 1 is the ranking for a given entry
    gold_ranks = gold_ranks[1]
    mean_rank = (gold_ranks + 1).type(torch.float).mean()

    recall_val = {k: ((gold_ranks < k).sum().type(torch.float) / number_entries) for k in recall_at if
                  k <= number_entries}

    #TODO Include a variant where there is threshold to consider.
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
