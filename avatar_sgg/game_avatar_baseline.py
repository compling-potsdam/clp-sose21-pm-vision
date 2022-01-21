"""
    Avatar action routines
"""

from avatar_sgg.config.util import get_config
from avatar_sgg.game_avatar_abstract import Avatar
from avatar_sgg.captioning.catr.inference import CATRInference
from sentence_transformers import SentenceTransformer
import random
import os
from avatar_sgg.image_retrieval.evaluation import vectorize_captions


class BaselineAvatar(Avatar):
    """
        The Baseline Avatar, using captioning models and BertSentences for Image Retrieval
    """

    def __init__(self, image_directory):
        config = get_config()
        sentence_bert_device = config["captioning"]["sentence_bert"]["cuda_device"]
        sentence_bert_model = config["captioning"]["sentence_bert"]["model"]
        config = config["game_setup"]
        self.debug = config["debug"]

        config = config["avatar"]

        self.max_number_of_interaction = config["max_number_of_interaction"]

        self._print(f"The avatar will allow only {self.max_number_of_interaction} interactions with the human player.")

        self.number_of_interaction = 0

        self.image_directory = image_directory
        self.observation = None
        self.map_nodes = None
        self.caption_expert: CATRInference = CATRInference()
        self.similarity_expert: SentenceTransformer = SentenceTransformer(sentence_bert_model).to(sentence_bert_device)
        self._print(f"Avatar using SentenceBert with {sentence_bert_model} for caption similarity.")
        self.similarity_threshold = config["similarity_threshold"]
        self.aggregate_interaction = config["aggregate_interaction"]
        self._print(f"Threshold for similarity based retrieval: {self.similarity_threshold}")
        self._print(f"Aggregate Interaction: {self.aggregate_interaction}")
        self.generated_captions = {}
        self.map_nodes_real_path = {}
        self.vectorized_captions = None
        self.interactions = []

    def is_interaction_allowed(self):
        """
        check if the avatar is still allowed to process messages.
        :return:
        """
        return self.number_of_interaction < self.max_number_of_interaction

    def get_prediction(self):
        """
        Should return the room identified by the avatar
        :return:
        """
        # TODO Replace it by the results of the models in use.
        pass
        choice = random.choice(list(self.map_nodes.items()))
        return choice

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
        self.map_nodes = map_nodes
        self.__init_captions()

    def __init_captions(self):
        """
        Generate captions and vector representation for them.
        :return:
        """
        self.map_nodes_real_path = {k: { "physical_path" : os.path.join(self.image_directory, self.map_nodes[k])} for k in self.map_nodes.keys() }
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

    def __generate_response(self, message: str) -> str:
        self.__increment_number_of_interaction()
        message = message.lower()
        self.interactions.append(message)

        if message.startswith("what"):
            if self.observation:
                return "I see " + self.observation["image"]
            else:
                return "I dont know"

        if message.startswith("where"):
            if self.observation:
                return "I can go " + self.directions_to_sent(self.observation["directions"])
            else:
                return "I dont know"

        if message.endswith("?"):
            if self.observation:
                return "It has maybe something to do with " + self.observation["image"]
            else:
                return "I dont know"

        return f"You interacted {self.number_of_interaction} times with me."

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
