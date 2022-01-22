import string


class Avatar(object):
    """
        The Abstract Avatar Class
    """

    def __init__(self):
        self.debug = True
        self.DIRECTION_TO_WORD = {
            "n": "north",
            "e": "east",
            "w": "west",
            "s": "south"
        }
        self.interactions = []
        self.room_found = False
        self.number_of_interaction = None
        self.current_candidate_similarity = None

    def get_number_of_interaction(self):
        return self.number_of_interaction

    def get_current_candidate_similarity(self):
        return self.current_candidate_similarity

    def direction_to_word(self, direction: str):
        if direction in self.DIRECTION_TO_WORD:
            return self.DIRECTION_TO_WORD[direction]
        return direction

    def directions_to_sent(self, directions: str):
        if not directions:
            return "nowhere"
        n = len(directions)
        if n == 1:
            return self.direction_to_word(directions[0])
        words = [self.direction_to_word(d) for d in directions]
        return ", ".join(words[:-1]) + " or " + words[-1]

    def _print(self, *message):
        """
        Print your message when debug is true
        :param self:
        :param message:
        :return:
        """
        if self.debug:
            print(*message)

    def step(self, observation: dict) -> dict:
        """
        The current observation for the avatar_sgg.

        For new player messages only the 'message' will be given.
        For new situations the 'image' and 'directions' will be given.

        The agent should return a dict with "move" or "response" keys.
        The response will be sent to the player.
        The move command will be executed by the game master.
        Possible move commands are: {"n": "north", "e": "east", "w": "west", "s": "south"}

        :param observation: {"image": str, "directions": [str], "message": str }
        :return: a dict with "move" and/or "response" keys; the dict could also be empty to do nothing
        """
        raise NotImplementedError("step")

    def aggregate_interactions(self):

        cleaned_messages = [i if i[-1] in string.punctuation else i + "." for i in self.interactions]
        return " ".join(cleaned_messages)

    def is_interaction_allowed(self):
        """
        Depends on the number of interactions allowed per game
        :return:
        """
        raise NotImplementedError("is_interaction_allowed")

    def reset(self):
        """
        Reset the avatar attributes before starting a new game.
        :return:
        """
        raise NotImplementedError("reset")

    def get_prediction(self):
        """
        Should return the room identified by the avatar
        :return:
        """
        raise NotImplementedError("get_prediction")
