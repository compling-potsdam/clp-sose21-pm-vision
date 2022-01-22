"""
    Slurk client as wrapper for the avatar_sgg agent to handle the slurky socketio stuff
"""
import socketIO_client, random

from avatar_sgg.game_avatar import Avatar


def check_error_callback(success, error=None):
    if error:
        print("Error: ", error)


class AvatarBot(socketIO_client.BaseNamespace):
    """
        Listens to the users messages and execute move commands. Kind of an environment adapter.

        The game rooms are already created with the setup script. We only join the rooms.

        The players might be already in the game room or join later.
    """
    ACTIONS = ["move", "respond"]
    NAME = "Avatar"

    def __init__(self, io, path):
        super().__init__(io, path)
        self.id = None
        self.agent = None
        self.emit("ready")  # invokes on_joined_room for the token room so we can get the user.id
        self.active = True

    def set_agent(self, agent: Avatar):
        self.agent = agent

    def on_joined_room(self, data: dict):
        """
        Send to the user itself to get the user_id.

        :param data: {'room': room.name, 'user': user.id}
        """
        print("on_joined_room", data)
        if not self.id:
            self.id = data['user']

    def on_text_message(self, data: dict):
        """
            This is a bit complicated, because we have two sender of events:
                A. the game master, when a new game observation or game reward is triggered
                B. the play who is sending messages directly to the bot (without the observation)

                Actually, it would be cleaner to always get the observation
                or rather the player message as part of the observation / environment.

            :param data:
            A. for observations it looks like {
                "type": room_type,
                "instance": game_obs["descriptors"]["instance"],
                "situation": "This is what you see. You can go %s." % (directions_to_sent(directions)),
                "player": player,
                "directions": directions
            }
            B. for messages it looks like {
                'msg': payload['msg'],
                'user': {
                    'id': current_user_id,
                    'name': current_user.name,
                },
                'room': room.name
                'timestamp': timegm(datetime.now().utctimetuple()),
                'private': False (doesnt really matter here),
                'html': payload.get('html', False)
            }
        """
        if not self.id:
            return  # not ready yet
        # print( "on_text_message", data)
        message = data["msg"]
        room_name = data["room"]
        user_name = data["user"]["name"]

        if user_name == "Game Master":
            # A. Handle new observations from game master (initially, at the end or after a move command)

            is_dict = isinstance(message, dict)

            if is_dict and "observation" in message:
                obs = message["observation"]
                actions = self.agent.step({"image": obs["instance"],
                                           "directions": obs["directions"],
                                           "message": None,
                                           "reward": obs["reward"],
                                           "done": obs["done"]})
                self.__perform_actions(actions, room_name)
            elif is_dict and "map_nodes" in message:
                self.active = True
                # only happens at the start of anew round, so the avatar states need reset.
                print("Avatar state reset.")
                self.agent.reset()
                print("map nodes set on agent:", message["map_nodes"])
                self.agent.set_map_nodes(message["map_nodes"])

            return  # ignore other game master messages for the bot

        user_id = data["user"]["id"]
        if user_id == self.id:
            return  # ignore our own messages

        # B. Handle player messages (player cannot send anything else than messages)
        actions = self.agent.step({"image": None,
                                   "directions": None,
                                   "message": message,
                                   "reward": None,
                                   "done": None})
        self.__perform_actions(actions, room_name)

    def __perform_actions(self, actions, room_name):
        if self.active:
            # TODO This is where you will perform the image similarity operation on the stored gallery.

            # {'command': 'done', 'user': {'id': 3, 'name': 'Player 1'}, 'room': 'avatar_room'}
            if not self.agent.is_interaction_allowed():
                self.__send_done_command(room_name)

            if "move" in actions:
                command = actions["move"]
                self.__send_command(command, room_name)
            if "response" in actions:
                response = actions["response"]
                self.__send_message(response, room_name)

    def __send_message(self, message, room_name):
        self.emit("text", {'room': room_name, 'msg': message}, check_error_callback)

    def __send_done_command(self, room_name):
        self.active = False
        # The agent finishes the game after sufficient number of interactions
        guessed_room = self.agent.get_prediction()
        number_interactions = self.agent.get_number_of_interaction()
        current_candidate_similarity = self.agent.get_current_candidate_similarity()
        self.emit("message_command", {'room': room_name,
                                      'command': {"msg": "done",
                                                  'guessed_room': guessed_room,
                                                  'number_of_interaction': number_interactions,
                                                  'current_candidate_similarity': current_candidate_similarity
                                                  }},
                  check_error_callback)

    def __send_command(self, command, room_name):
        self.emit("message_command", {'room': room_name, 'command': command}, check_error_callback)
