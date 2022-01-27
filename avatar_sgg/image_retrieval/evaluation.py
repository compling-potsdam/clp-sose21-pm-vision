import torch, collections
from sentence_transformers import SentenceTransformer
from avatar_sgg.sentence_embedding.util import vectorize_captions
from avatar_sgg.dataset.util import get_categories
from avatar_sgg.captioning.catr.inference import CATRInference
import string
import json
import os

def calculate_normalized_cosine_similarity_for_captions(input):
    """
    Input has the dimensions: number of entries * 2 * vector dimension
    :param input:
    :return:
    """

    first_embeds = input[:, 0, :]
    second_embeds = input[:, 1, :]

    return calculate_normalized_cosine_similarity(first_embeds, second_embeds)


def calculate_normalized_cosine_similarity(gallery_input, query):
    """
    Input has the dimensions: number of entries X * vector dimension
    query has the dimension number of entries Y * vector dimension
    :param input:
    :return:
    """

    # Trivial check to insure the dimension stated in the method header.
    num_dim_gallery = len(gallery_input.shape)
    num_dim_query = len(query.shape)
    assert num_dim_gallery < 3
    assert num_dim_query < 3
    if num_dim_query == 1:
        query = query.unsqueeze(0)
    if num_dim_gallery == 1:
        gallery_input = gallery_input.unsqueeze(0)
    assert gallery_input.shape[1] == query.shape[1]

    similarity = gallery_input @ query.T

    norm_1 = gallery_input.norm(dim=1, p=2)
    norm_2 = query.norm(dim=1, p=2)
    norm = norm_1.unsqueeze(1) * norm_2.unsqueeze(1).T
    # Normalize the similarity scores to make them comparable
    similarity = similarity / norm

    return similarity


def compute_recall_johnson_feiefei(similarity, threshold, category, recall_at: list = [1, 2, 3, 4, 5, 10, 20, 50, 100]):
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

    recall_val = {"recall_at_" + str(k): ((gold_ranks < k).sum().type(torch.float) / number_entries).to("cpu").numpy()
                  for k in recall_at if
                  k <= number_entries}

    mean_rank = mean_rank.to("cpu").numpy()

    return recall_val, mean_rank


def compute_recall_on_category(similarity, threshold, category, recall_at: list = [1, 2, 3, 4, 5, 10, 20, 50, 100]):
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
    recall_val = {"recall_at_" + str(k): (numerator(k) / denominator(k)).to("cpu").numpy() for k in recall_at if
                  k <= number_entries}
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

    similarity = calculate_normalized_cosine_similarity_for_captions(stacked_vectors)
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

    similarity_caption_1 = calculate_normalized_cosine_similarity_for_captions(comparison_1)
    similarity_caption_2 = calculate_normalized_cosine_similarity_for_captions(comparison_2)
    recall_val, mean_rank = recall_funct(similarity_caption_1, threshold, category)
    recall_val_2, mean_rank_2 = recall_funct(similarity_caption_2, threshold, category)

    print(f"Threshold for retrieval: {threshold}")
    for k in recall_val.keys():
        print(f"Average {k}: {(recall_val[k] + recall_val_2[k]) / 2}")

    average_mean_rank = ((mean_rank + mean_rank_2) / 2)
    recall_val["mean_rank"] = average_mean_rank

    print(f"Average Mean Rank: {average_mean_rank}")
    average_similarity = ((similarity_caption_1.diag().mean() + similarity_caption_2.diag().mean()) / 2).to(
        "cpu").numpy()
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

            data_split[path]["caption"] = [human_captions, catr_caption]


def use_merged_sequence(data_split):
    """
    Remove human captions, use the merged sequences of descriptions instead.
    Intended to use when CATR caption have been generated
    :param data_split:
    :return:
    """

    for path in data_split.keys():
        range = len(data_split[path]["caption"])
        catr_caption = data_split[path]["caption"][range - 1]
        data_split[path]["caption"] = [data_split[path]["merged_sequences"], catr_caption]


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

def read_game_logs(file_path):
    """
    It returns a dictionary where each key holds information for a particular (finished) game session.
    General statistics about the game session are provided: score, the number of questions asked by the player,
    the number of orders and whole messages during the game session
    :param file_path:
    :return:
    """

    if os.path.isfile(file_path):
        with open(file_path, "r") as read_file:
            log = json.load(read_file)
        # event_type = set([e["event"] for e in log ])
        # the event types: command, text_message, set_attribute, join
        # print("event types", event_type)

        # sort all messages chronologically
        log.sort(key=lambda x: x["date_modified"])

        start = None
        end = None
        real_end = None  # WHen The came master says COngrats or you die, because rest of the messages looks like bugs...
        episode_list = []
        length = len(log)
        game_finished = False
        # Episode are being searched between 2 starts commands
        # only the one where the command done has been issued is kept
        for i, l in enumerate(log):
            if "command" in l.keys():
                if l["command"] == "start":
                    if start == None:
                        start = i
                    elif end == None:
                        end = i
                if type(l["command"]) is dict:
                    game_finished = True

            if l["user"]["id"] == 1 and l["event"] == "text_message" and type(l["message"]) is str and (
                    l["message"].startswith("Congrats") or l["message"].startswith(
                "The rescue robot was not able to identify your location.")):
                real_end = i + 1  # +1 because we want to include this message in the log slice...
            if start is not None and end is not None:
                if game_finished:
                    episode_list.append(log[start:real_end])
                start = end
                end = None
                real_end = None
                game_finished = False

            if i + 1 == length:
                if start is not None and end is None and game_finished:
                    episode_list.append(log[start:real_end])

        score_list = {}
        # Within each episode, these are the only 2 entries that I care about:
        #
        # {'current_candidate_similarity': 0.6925256848335266, 'guessed_room': 'k/kitchen/ADE_train_00010362.jpg', 'msg': 'done', 'number_of_interaction': 2}
        # {'map_nodes': {'0': 'p/parking_garage/outdoor/ADE_train_00015105.jpg', '1': 'k/kitchen/ADE_train_00010362.jpg', '2': 'k/kitchen/ADE_train_00010338.jpg', '3': 'r/restroom/outdoor/ADE_train_00015844.jpg', '4': 'j/jacuzzi/indoor/ADE_train_00009927.jpg', '5': 's/shower/ADE_train_00016282.jpg', '6': 'r/restroom/outdoor/ADE_train_00015844.jpg', '7': 'j/jacuzzi/indoor/ADE_train_00009928.jpg'}, 'start_game': "You are a rescue bot. A person is stuck and needs its medicine to survive. I'm afraid, you don't have a human detector attached, so the other one has to decide, if you can recognize the location out of a list of room. Therefore listen carefully to the instructions..."}
        # Optional: what the human player inputs.
        for i, e in enumerate(episode_list):
            # the number of answers the avatar utters gives us the number of question asked
            # num_questions = sum(
            #     [1 for m in e if m["user"]["name"] == "Avatar" and m["event"] == "text_message"])

            sent_game_map = [m["data"]["message"]["map_nodes"] for m in e if "data" in m.keys() and "message" in m["data"].keys() and type(m["data"]["message"]) is dict and "map_nodes" in m["data"]["message"].keys()][0]

            # Just sum every messages ending with a question mark issueed by the user...
            num_questions = sum([1 for m in e if m["user"]["name"] != "Avatar" and m["user"]["id"] != 1 and m[
                "event"] == "text_message" and type(m["message"]) is str and m["message"].endswith("?")])

            # user id 1 is alway the game master, we are looping here on the messages of the "real" player
            # when we tell the avatar to change location, we don't get an answer, this is why the substraction gives the number of orders
            # this does not include the order "done"
            # num_orders = sum(
            #     [1 for m in e if m["user"]["name"] != "Avatar" and m["user"]["id"] != 1 and m[
            #         "event"] == "text_message"]) - num_questions

            # Just sum every order of type "go west". Describe orders are not counted.
            num_orders = sum([1 for m in e if m["user"]["name"] != "Avatar" and m["user"]["id"] != 1 and m[
                "event"] == "text_message" and type(m["message"]) is str and (
                                      "east" in m["message"].lower() or "north" in m["message"].lower() or "west" in m[
                                  "message"].lower() or "south" in m["message"].lower() or "back" in m["message"].lower())])

            game_won = sum([1 for m in e if m["user"]["id"] == 1 and m[
                "event"] == "text_message" and type(m["message"]) is str and m["message"].startswith("Congrats")]) > 0

            # Work-Around - the final reward giving +1.0 on success and -1.0 on loss happens after the messages
            # Saying "congratulations" or "you die horribly" just repeating the message when the game starts.
            # We had to exclude that message to segment finished games but this is why we have to add these rewards here manually...

            final_reward = -1.0
            if game_won:
                final_reward = 1.0
            score_list[i] = {"score": sum([m["message"]["observation"]["reward"] for m in e if
                                           "message" in m.keys() and type(m["message"]) is dict])+final_reward,
                             "num_questions": num_questions, "num_orders": num_orders, "game_session": e,
                             "game_won": game_won}

        return score_list

    else:
        raise Exception(f"{file_path} is not a correct file path.")


def test_cosine():
    from avatar_sgg.config.util import get_config
    from avatar_sgg.dataset.util import get_ade20k_split

    output_dir = os.path.join(get_config()["output_dir"], "image_retrieval")
    train, dev, test = get_ade20k_split()

    current = dev
    vectorizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    stacked_vectors = vectorize_captions(current, vectorizer)
    first_embeds = stacked_vectors[:, 0, :]
    query = stacked_vectors[0, 1, :]
    similarity = calculate_normalized_cosine_similarity(first_embeds, query)
    values, ranks = torch.topk(similarity, 1, dim=0)
    print(values, ranks)

if __name__ == "__main__":
    print("Start")
    #test_cosine()
    log_file = "/home/rafi/PycharmProjects/clp-sose21-pm-vision/results/slurk_logs/sgg_test.txt"
    read_game_logs(log_file)
    print("End")
