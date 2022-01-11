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

def compute_recall(similarity, threshold=None, recall_at: list = [1, 2, 5, 8, 10, 20]):

    if not threshold:
        # The mean of the diagonal is the criteria to define something "meaningful"
        pred_rank = (similarity >= similarity.diag().mean().view(-1, 1)).sum(-1)
    else:
        pred_rank = (similarity >= (torch.ones_like(similarity.diag()) * threshold).view(-1, 1)).sum(-1)
    num_sample = pred_rank.shape[0]
    for k in recall_at:
        if k <= num_sample:
            # It's weird the way recall is computed (https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/).
            # The only way that makes sense is the following:
            # For each image, we know what is the expected best result for a given text query.
            # The only sensible way to do that would be, if we consider the best k results as in @K metrics,
            # is our image included in? We can then compute a proportion of times our system includes the true image in
            # its top k.
            # This is how I understood anyhow https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf
            # p.6
            # TODO Change this!
            print('Recall @ %d: %.4f; ' % (k, float((pred_rank < k).sum()) / num_sample))
