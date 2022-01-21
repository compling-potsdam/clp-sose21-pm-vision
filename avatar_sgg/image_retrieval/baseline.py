from avatar_sgg.dataset.util import get_ade20k_split
from avatar_sgg.config.util import get_config
from avatar_sgg.image_retrieval.evaluation import compute_similarity, compute_average_similarity, \
    compute_recall_on_category, compute_recall_johnson_feiefei, add_inferred_captions, merge_human_captions, \
    use_merged_sequence
import numpy as np
import pandas as pd
import os


def run_evaluation(evaluation_name, split, similarity_function, threshold_list, recall_function, output_dir):
    values = []
    print(f"############## Start Evaluation: {evaluation_name} ############## ")
    for t in threshold_list:
        print("\n")
        print(f"Threshold: {t}")
        val = similarity_function(split, t, recall_function)
        values.append(val)
        print("\n")
    print(f"############## End Evaluation: {evaluation_name} ############## ")
    df = pd.DataFrame(values)
    output_path = os.path.join(output_dir, evaluation_name + ".csv")
    print(f"Saving data to {output_path}")
    df.to_csv(output_path)


if __name__ == "__main__":
    print("Start")
    output_dir = os.path.join(get_config()["output_dir"], "image_retrieval")

    train, dev, test = get_ade20k_split()

    current = test
    threshold_list = [None]
    # This range has been chosen because the mean of the diagonal on the dev set was around 0.6X
    threshold_list.extend(np.linspace(0.55, 0.7, 15))

    eval_name = lambda caption_type, recall_type: f"{caption_type}_{recall_type}"
    human_caption = "human_captions_query"
    fei_fei_recall = "feifei_johnson_recall"
    catr_caption = "catr_captions_query"
    merged_human_caption = "merged_human_caption_catr_captions_query"
    merged_sequences_captions = "merged_sequences_catr_captions_query"
    ade20k_category_recall = "ade20k_category_recall"

    evaluation_name = eval_name(human_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    evaluation_name = eval_name(human_caption, ade20k_category_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_on_category, output_dir)

    add_inferred_captions(current)
    evaluation_name = eval_name(catr_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_average_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    evaluation_name = eval_name(catr_caption, ade20k_category_recall)
    run_evaluation(evaluation_name, current, compute_average_similarity, threshold_list, compute_recall_on_category,
                   output_dir)

    merge_human_captions(current)
    evaluation_name = eval_name(merged_human_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    evaluation_name = eval_name(merged_human_caption, ade20k_category_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_on_category, output_dir)

    evaluation_name = eval_name(merged_human_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    evaluation_name = eval_name(merged_human_caption, ade20k_category_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_on_category, output_dir)

    use_merged_sequence(current)
    evaluation_name = eval_name(merged_sequences_captions, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    evaluation_name = eval_name(merged_sequences_captions, ade20k_category_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_on_category, output_dir)

    evaluation_name = eval_name(merged_sequences_captions, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)

    evaluation_name = eval_name(merged_sequences_captions, ade20k_category_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_on_category, output_dir)


    print("Done")
