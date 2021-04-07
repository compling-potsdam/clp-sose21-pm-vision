"""
    Slurk client for the game master
"""
import socketIO_client

from avatar.game import MapWorldGame


def check_error_callback(success, error=None):
    if error:
        print("Error: ", error)


class GameMaster(socketIO_client.BaseNamespace):
    """
        Coordinates player between games.

        The game rooms are already created with the setup script. We only join the rooms.
    """

    NAME = "Game Master"

    def __init__(self, io, path):
        super().__init__(io, path)
        self.id = None
        self.base_image_url = None
        self.games = {}  # game by room_name
        self.emit("ready")  # invokes on_joined_room for the token room so we can get the user.id

    def set_base_image_url(self, base_image_url: str):
        self.base_image_url = base_image_url

    def on_joined_room(self, data: dict):
        """
        Send to the user itself.

        :param data: {'room': room.name, 'user': user.id}
        """
        print("on_joined_room", data)
        if not self.id:
            self.id = data['user']
        room_name = data["room"]
        # We prepare a game for each room we join
        self.games[room_name] = MapWorldGame(room_name)
        # TODO if there are already users in the room, join them to the game

    def on_command(self, data: dict):
        """
        :param data: {
                'command': payload['command'],
                'user': {
                    'id': current_user_id,
                    'name': current_user.name,
                },
                'room': room.name
                'timestamp': timegm(datetime.now().utctimetuple()),
                'private': False (commands cannot be private when coming from the standard layout)
            }
        """
        game: MapWorldGame = self.games[data["room"]]
        command = data["command"]
        user_id = data["user"]["id"]
        if game.is_avatar(user_id):
            self.__step_game(command, data["user"], game)
        else:
            self.__control_game(command, data["user"], game)

    def __step_game(self, command: str, user: dict, game: MapWorldGame):
        """
            Actions performed by the avatar.
        """
        observation = game.step(user["id"], command)
        self.__send_observation(observation, game.room, user["id"])

    def __control_game(self, command: str, user: dict, game: MapWorldGame):
        """
            Actions performed by the player.
        """
        if command == "done":
            self.__end_game_if_possible(user, game)
        elif command == "restart":
            self.__start_game_if_possible(user, game)
        else:
            self.__send_private_message(
                "You cannot use the command '/%s'. You may use '/done' or '/restart'." % command,
                game.room, user["id"])

    def on_status(self, data: dict):
        """
        Send to room participants if a user joins or leaves the room.

        We expect that the game master is the first person in the game room.

        :param data: {
            'type': 'join',
            'user': {
                'id': current_user.id,
                'name': current_user.name,
            },
            'room': room.name,
            'timestamp': timegm(datetime.now().utctimetuple())
        }
        """
        print("on_status", data)
        user = data["user"]
        room_name = data["room"]
        if user["id"] == self.id:
            return
        game: MapWorldGame = self.games[room_name]
        if data["type"] == "join":
            game.join(user["id"], user["name"])
            self.__send_private_message(f"Welcome, {user['name']}, I am your {GameMaster.NAME}!", game.room, user["id"])
            self.__start_game_if_possible(user, game)
        if data["type"] == "leave":
            self.__pause_game(user, game)

    def __pause_game(self, user: dict, game: MapWorldGame):
        self.__send_room_message(f"{user['name']} left the game.", game.room)

    def __end_game_if_possible(self, user: dict, game: MapWorldGame):
        if game.is_done():
            self.__send_private_message(
                "Congrats! You are correct. The avatar has reached your room. Type /restart if you want to play again.",
                game.room, user["id"])
        else:
            self.__send_private_message("Nope, the avatar has not reached your room yet.", game.room, user["id"])

    def __start_game_if_possible(self, user: dict, game: MapWorldGame):
        if game.is_ready():
            self.__start_game(game)
        else:
            self.__send_private_message("I will prepare the game for you now... this might take a while.!",
                                        game.room, user["id"])

    def __start_game(self, game: MapWorldGame):
        game.reset(4, 4, 8)
        user_ids = game.get_players()
        for user_id in user_ids:
            self.__send_private_message("The game starts now... Have fun!", game.room, user_id)
            # Send initial observations
            observation = game.get_observation(user_id)
            self.__send_observation(observation, game.room, user_id)
            # Send initial mission statements
            self.__send_private_message(game.get_mission(user_id), game.room, user_id)

    def __send_observation(self, observation: dict, room_name: str, user_id: int):
        if "instance" in observation:
            self.__send_private_image(observation["instance"], room_name, user_id)
        if "situation" in observation:
            self.__send_private_message(observation["situation"], room_name, user_id)

    def __send_room_message(self, message: str, room_name: str):
        self.emit("text", {"msg": message, "room": room_name}, check_error_callback)

    def __send_private_message(self, message: str, room_name: str, user_id: int):
        self.emit("text", {"msg": message, "room": room_name, "receiver_id": user_id}, check_error_callback)

    def __send_private_image(self, image_url: str, room_name: str, user_id: int):
        image_url = f"{self.base_image_url}/{image_url}"
        print(image_url)
        self.emit("image", {"url": image_url, "room": room_name, "receiver_id": user_id}, check_error_callback)
