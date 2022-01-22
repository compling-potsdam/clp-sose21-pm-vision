"""
    Avatar action routines
"""

from avatar_sgg.config.util import get_config
from avatar_sgg.game_avatar_abstract import Avatar
from avatar_sgg.captioning.catr.inference import CATRInference
from sentence_transformers import SentenceTransformer
import random
import os
from avatar_sgg.image_retrieval.evaluation import vectorize_captions, calculate_normalized_cosine_similarity
from collections import defaultdict
import torch


class BaselineAvatar(Avatar):
    """
        The Baseline Avatar, using captioning models and BertSentences for Image Retrieval
    """

    def __init__(self, image_directory):
        config = get_config()
        if image_directory is None:
            image_directory = os.path.join(config["ade20k"]["root_dir"], "images", "training")
        sentence_bert_device = config["captioning"]["sentence_bert"]["cuda_device"]
        sentence_bert_model = config["captioning"]["sentence_bert"]["model"]
        config = config["game_setup"]
        self.debug = config["debug"]

        config = config["avatar"]

        self.max_number_of_interaction = config["max_number_of_interaction"]

        self._print(f"The avatar will allow only {self.max_number_of_interaction} interactions with the human player.")

        self.image_directory = image_directory

        self.caption_expert: CATRInference = CATRInference()
        self.similarity_expert: SentenceTransformer = SentenceTransformer(model_name_or_path=sentence_bert_model, device=sentence_bert_device)
        self._print(f"Avatar using SentenceBert with {sentence_bert_model} for caption similarity.")
        self.similarity_threshold = config["similarity_threshold"]
        self.minimum_similarity_threshold = config["minimum_similarity_threshold"]
        self.aggregate_interaction = config["aggregate_interaction"]
        self._print(f"Threshold for similarity based retrieval: {self.similarity_threshold}")
        self._print(f"Aggregate Interaction: {self.aggregate_interaction}")
        self.number_of_interaction = None
        self.observation = None
        self.map_nodes = None
        self.generated_captions = None
        self.map_nodes_real_path = None
        self.vectorized_captions = None
        self.vectorized_interactions = None
        self.current_candidate_similarity = None
        self.current_candidate_rank = None

        self.reset()

    def reset(self):
        """
        Reset important attributes for the avatar.
        :return:
        """
        self.number_of_interaction = 0
        self.observation = None
        self.map_nodes = None
        self.generated_captions = {}
        self.map_nodes_real_path = {}
        self.vectorized_captions = None
        self.vectorized_interactions = []
        self.current_candidate_similarity = 0.0
        self.current_candidate_rank = None
        self.interactions = []
        self.room_found = False

    def is_interaction_allowed(self):
        """
        check if the avatar is still allowed to process messages.
        :return:
        """

        if self.room_found:
            return False

        return (self.number_of_interaction < self.max_number_of_interaction)

    def get_prediction(self):
        """
        Should return the room identified by the avatar
        :return:
        """
        prediction = None

        if self.current_candidate_rank is not None:
            prediction = self.map_nodes[self.current_candidate_rank]

        # choice = random.choice(list(self.map_nodes.items()))
        return prediction

    def __increment_number_of_interaction(self):
        self.number_of_interaction += 1

    def set_map_nodes(self, map_nodes: dict):
        """
        Only called once, when the labyrinth is initialized.
        example of entry in map_nodes:
        0: 'w/waiting_room/ADE_train_00019652.jpg'
        :param map_nodes:
        :return:
        """
        # As dictionary is sent with socket io, the int keys were converted into string.
        self.map_nodes = {int(k): map_nodes[k] for k in map_nodes.keys()}
        self.__init_captions()

    def __init_captions(self):
        """
        Generate captions and vector representation for them.
        :return:
        """
        self.map_nodes_real_path = {k: {"physical_path": os.path.join(self.image_directory, self.map_nodes[k])} for k in
                                    self.map_nodes.keys()}
        for k in self.map_nodes_real_path.keys():
            generated_caption = self.caption_expert.infer(self.map_nodes_real_path[k]["physical_path"])
            self.map_nodes_real_path[k]["caption"] = generated_caption
        self.vectorized_captions = vectorize_captions(self.map_nodes_real_path, self.similarity_expert)

    def step(self, observation: dict) -> dict:
        if self.debug:
            print(observation)
        actions = dict()
        if observation["image"]:
            self.__update_observation(observation)
        if observation["message"]:
            self.__update_actions(actions, observation["message"])
        return actions

    def __update_observation(self, observation: dict):
        self.observation = observation

    def __update_actions(self, actions, message):
        if "go" in message.lower():
            actions["move"] = self.__predict_move_action(message)
        else:
            actions["response"] = self.__generate_response(message)

    def __set_room_found(self):

        if self.current_candidate_similarity >= self.similarity_threshold:
            self.room_found = True

    def __generate_response(self, message: str) -> str:
        self.__increment_number_of_interaction()
        message = message.lower()
        self.interactions.append(message)
        query_message = message
        if self.aggregate_interaction:
            query_message = self.aggregate_interactions()
        vectorized_query = self.similarity_expert.encode(query_message, convert_to_tensor=True)
        self.vectorized_interactions.append(vectorized_query)
        similarity = calculate_normalized_cosine_similarity(self.vectorized_captions, vectorized_query)
        values, ranks = torch.topk(similarity, 1, dim=0)
        values = values[0][0].to("cpu").numpy()
        ranks = ranks[0][0].to("cpu").numpy()
        if values > self.current_candidate_similarity:
            self.current_candidate_similarity = values
            self.current_candidate_ranks = ranks

        self.__set_room_found()

        found_msg = ""
        if self.room_found:
            found_msg = " I believe I found the room based on your description."
        return f"You interacted {self.number_of_interaction} times with me.{found_msg}"

    def __predict_move_action(self, message: str) -> str:
        if "north" in message:
            return "n"
        if "east" in message:
            return "e"
        if "west" in message:
            return "w"
        if "south" in message:
            return "s"
        return "nowhere"
