"""
    Avatar player models
"""
from avatar.game import directions_to_sent


class Avatar(object):
    """
        The avatar methods to be implemented
    """

    def look_at(self, observation: dict):
        """
        The current observation for the avatar.

        :param observation: {
            "type": room_type,
            "instance": game_obs["descriptors"]["instance"],
            "situation": "This is what you see. You can go %s." % (directions_to_sent(directions)),
            "player": player,
            "directions": directions
        }
        """
        raise NotImplementedError("look_at")

    def read_and_decide(self, message: str) -> str:
        """
            Read the message and decide whether to "respond" or to "move".

            If "move", then "read_and_move" will be called.

            If "respond", then "read_and_respond" will be called.

            Otherwise, nothing will happen.

            :return: either "respond" or "move"
        """
        raise NotImplementedError("read_and_decide")

    def read_and_respond(self, message: str) -> str:
        """
            Read the message and generate a response. The response will be sent to the player.

            :return: a response message
        """
        raise NotImplementedError("read_and_respond")

    def read_and_move(self, message: str) -> str:
        """
            Read the message and generate a command. The command will be executed by the game master.

            Possible commands are: {"n": "north", "e": "east", "w": "west", "s": "south"}

            :return: a command message
        """
        raise NotImplementedError("read_and_move")


class SimpleAvatar(Avatar):
    """
        The simple avatar is only repeating the observations.
    """

    def __init__(self):
        self.observation = None

    def look_at(self, observation: dict):
        self.observation = observation

    def read_and_decide(self, message: str) -> str:
        if message is None:
            return "ignore"
        if "go" in message.lower():
            return "move"
        else:
            return "respond"

    def read_and_respond(self, message: str) -> str:
        message = message.lower()
        if message.startswith("what"):
            return "I see " + self.observation["instance"]
        if message.startswith("where"):
            return "I can go " + directions_to_sent(self.observation["directions"])
        if message.endswith("?"):
            return "It has maybe something to do with " + self.observation["type"]
        return "I do not understand"

    def read_and_move(self, message: str) -> str:
        if "north" in message:
            return "n"
        if "east" in message:
            return "e"
        if "west" in message:
            return "w"
        if "south" in message:
            return "s"
        return "nowhere"
