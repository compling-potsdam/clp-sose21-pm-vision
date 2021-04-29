"""
    Avatar action routines
"""

DIRECTION_TO_WORD = {
    "n": "north",
    "e": "east",
    "w": "west",
    "s": "south"
}


def direction_to_word(direction: str):
    if direction in DIRECTION_TO_WORD:
        return DIRECTION_TO_WORD[direction]
    return direction


def directions_to_sent(directions: str):
    if not directions:
        return "nowhere"
    n = len(directions)
    if n == 1:
        return direction_to_word(directions[0])
    words = [direction_to_word(d) for d in directions]
    return ", ".join(words[:-1]) + " or " + words[-1]


class Avatar(object):
    """
        The avatar methods to be implemented
    """

    def step(self, observation: dict) -> dict:
        """
        The current observation for the avatar.

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


class SimpleAvatar(Avatar):
    """
        The simple avatar is only repeating the observations.
    """

    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.observation = None

    def step(self, observation: dict) -> dict:
        print(observation)  # for debugging
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
        message = message.lower()

        if message.startswith("what"):
            if self.observation:
                return "I see " + self.observation["image"]
            else:
                return "I dont know"

        if message.startswith("where"):
            if self.observation:
                return "I can go " + directions_to_sent(self.observation["directions"])
            else:
                return "I dont know"

        if message.endswith("?"):
            if self.observation:
                return "It has maybe something to do with " + self.observation["image"]
            else:
                return "I dont know"

        return "I do not understand"

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
