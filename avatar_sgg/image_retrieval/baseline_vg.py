from avatar_sgg.dataset.util import get_scene_graph_splits, get_scene_graph_loader
from avatar_sgg.config.util import get_config
from avatar_sgg.image_retrieval.evaluation import compute_similarity, compute_average_similarity, \
    compute_recall_on_category, compute_recall_johnson_feiefei, add_inferred_captions, merge_human_captions, \
    use_merged_sequence, run_evaluation

import numpy as np
import os




if __name__ == "__main__":
    print("Start")
    output_dir = os.path.join(get_config()["output_dir"], "image_retrieval")

    batch_size = 500
    train_ids, test_ids = get_scene_graph_splits()

    use_test = True
    use_val = False
    loader = get_scene_graph_loader(batch_size, train_ids, test_ids, test_on=use_test, val_on=use_val)
    current = next(enumerate(loader))


    threshold_list = [None]
    # This range has been chosen because the mean of the diagonal on the dev set was around 0.6X
    threshold_list.extend(np.linspace(0.55, 0.7, 15))

    eval_name = lambda caption_type, recall_type: f"{caption_type}_{recall_type}"
    human_caption = "vg_human_captions_query"
    fei_fei_recall = "feifei_johnson_recall"
    catr_caption = "vg_catr_captions_query"
    merged_human_caption = "vg_merged_human_caption_catr_captions_query"


    evaluation_name = eval_name(human_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_average_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)


    add_inferred_captions(current)
    evaluation_name = eval_name(catr_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_average_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)


    merge_human_captions(current)
    evaluation_name = eval_name(merged_human_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)


    evaluation_name = eval_name(merged_human_caption, fei_fei_recall)
    run_evaluation(evaluation_name, current, compute_similarity, threshold_list, compute_recall_johnson_feiefei,
                   output_dir)


    use_merged_sequence(current)

    print("Done")
