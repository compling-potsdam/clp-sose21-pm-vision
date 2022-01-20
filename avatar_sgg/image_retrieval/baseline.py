from avatar_sgg.dataset.util import get_ade20k_split
from avatar_sgg.image_retrieval.evaluation import compute_similarity, compute_average_similarity, compute_recall_on_category, compute_recall_johnson_feiefei,calculate_normalized_cosine_similarity, add_inferred_captions
import numpy as np


def run_evaluation(evaluation_name, split, similarity_function, threshold_list, recall_function):

    print(f"############## Start Evaluation: {evaluation_name} ############## ")
    for t in threshold_list:
        print("\n")
        print(f"Threshold: {t}")
        similarity_function(split, t, recall_function)
        print("\n")
    print(f"############## End Evaluation: {evaluation_name} ############## ")


if __name__ == "__main__":
    print("Start")
    train, dev, test = get_ade20k_split()

    current = test
    threshold_list = [None]
    #This range has been chosen because the mean of the diagonal on the dev set was around 0.6X
    threshold_list.extend(np.linspace(0.55, 0.7, 15))

    evaluation_name = "Based on human captions query only, Fei Fei / Johnson Recall"
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_johnson_feiefei)

    evaluation_name = "Based on human captions query only, ADE20K Category Based Recall"
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_on_category)

    add_inferred_captions(current)
    evaluation_name = "Based on CATR captions query, Fei Fei / Johnson Recall"
    run_evaluation(evaluation_name, current, compute_average_similarity, threshold_list, compute_recall_johnson_feiefei)

    evaluation_name = "Based on CATR captions query, ADE20K Category Based Recall"
    run_evaluation(evaluation_name, current, compute_average_similarity, threshold_list, compute_recall_on_category)

    print("Done")
