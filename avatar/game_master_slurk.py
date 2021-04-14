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

        The players might be already in the game room or join later.
    """

    NAME = "Game Master"

    def __init__(self, io, path):
        super().__init__(io, path)
        self.id = None
        self.base_image_url = None
        self.image_server_auth = None
        self.games = {}  # game by room_name
        self.map_width = 4
        self.map_height = 4
        self.map_rooms = 8
        self.emit("ready")  # invokes on_joined_room for the token room so we can get the user.id

    def set_base_image_url(self, base_image_url: str):
        self.base_image_url = base_image_url

    def set_image_server_auth(self, image_server_auth: str):
        self.image_server_auth = image_server_auth

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
        user = data["user"]

        # This might be useful, when the game master re-joins the room.
        # Then the players are in the game room before the game master.
        # The players are not in the game then, but must join again.
        if command == "rejoin":
            self.__join_game(user, game)
            return

        if game.is_avatar(user["id"]):
            self.__step_game(command, user, game)
        else:
            self.__control_game(command, user, game)

    def __step_game(self, command: str, user: dict, game: MapWorldGame):
        if game.is_done():
            self.__send_private_message(
                "The game already ended. Wait for the player to /restart the game.", game.room, user["id"])
            return
        """
            Actions performed by the avatar.
        """
        observation = game.step(user["id"], command)
        if observation:
            self.__send_observation(observation, game.room, user["id"])

    def __control_game(self, command: str, user: dict, game: MapWorldGame):
        """
            Actions performed by the player.
        """
        if command == "done":
            self.__end_game(user, game)
        elif command == "restart":
            self.__start_game_if_possible(user, game)
        elif command.startswith("set_map"):
            try:
                self.__set_map(command)
                self.__send_private_message(
                    "Set map to %s,%s,%s" % (self.map_width, self.map_height, self.map_rooms),
                    game.room, user["id"])
            except Exception as e:  # noqa
                self.__send_private_message(
                    "Wrong command '%s'. Should be like '/set_map:<width>,<height>,<rooms>'" % command,
                    game.room, user["id"])
        else:
            self.__send_private_message(
                "You cannot use the command '/%s'. You may use '/set_map', '/done', '/restart' or '/rejoin'." % command,
                game.room, user["id"])

    def __set_map(self, setting: str):
        """
        :param setting: str like "set_map:<width>,<height>,<rooms>"
        :exception Exception when format is wrong
        """
        parameters = setting.split(":")[1]
        parameters = [int(p.strip()) for p in parameters.split(",")]
        self.map_width = parameters[0]
        self.map_height = parameters[1]
        self.map_rooms = parameters[2]

    def on_status(self, data: dict):
        """
        Send to room participants if a user joins or leaves the room.

        We expect that the game master is the first person in the game room. Otherwise players have to use /rejoin.

        The mission statement is only send for game join events (so thats not repeating on restarts).

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
        game = self.games[room_name]
        if data["type"] == "join":
            self.__join_game(user, game)
        if data["type"] == "leave":
            self.__pause_game(user, game)

    def __join_game(self, user: dict, game: MapWorldGame):
        if game.has_player(user["id"]):
            # TODO log
            # self.__send_private_message(f"You already joined the game!", game.room, user["id"])
            return
        game.join(user["id"], user["name"])
        self.__send_private_message(f"Welcome, {user['name']}, I am your {GameMaster.NAME}!", game.room, user["id"])
        self.__send_private_message(game.get_mission(user["id"]), game.room, user["id"])
        self.__start_game_if_possible(user, game)

    def __pause_game(self, user: dict, game: MapWorldGame):
        # TODO log
        # self.__send_room_message(f"{user['name']} left the game.", game.room)
        pass

    def __end_game(self, user: dict, game: MapWorldGame):
        if game.is_done():
            self.__send_private_message(
                "The game already ended. Type /restart if you want to play again.", game.room, user["id"])
            return
        if game.is_success():
            self.__send_private_message(
                "Congrats, you'll survive! The rescue robot is at your location. "
                "Type /restart if you want to get lost again.", game.room, user["id"])
        else:
            self.__send_private_message("The rescue robot has not reached you. You die. Sorry. "
                                        "Type /restart if you want to get lost again.", game.room, user["id"])
        game.set_done()

    def __start_game_if_possible(self, user: dict, game: MapWorldGame):
        if game.is_ready():
            self.__start_game(game)
        else:
            self.__send_private_message("I will prepare the world for you now... this might take a while.!",
                                        game.room, user["id"])

    def __start_game(self, game: MapWorldGame):
        game.reset(self.map_width, self.map_height, self.map_rooms)
        user_ids = game.get_players()
        for user_id in user_ids:
            if game.is_avatar(user_id):
                self.__send_private_message(
                    "You are a rescue bot. A person is in need for your help. "
                    "I'm afraid, you don't have a human detector attached, so the other one has to decide, "
                    "if you reached the location. Therefore listen carefully to the instructions. Go -- have fun!",
                    game.room, user_id)
            else:
                self.__send_private_message(
                    "You are stranded, helpless. "
                    "You need to direct your invisible rescue robot to your location or you will die. Go -- have fun!",
                    game.room, user_id)
            # Send initial observations
            observation = game.get_observation(user_id)
            self.__send_observation(observation, game.room, user_id)

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
        if self.set_image_server_auth:
            image_url = image_url + f"?code={self.image_server_auth}"
        print(image_url)
        self.emit("set_attribute",
                  {"id": "current-image",
                   "attribute": "src",
                   "value": image_url,
                   "room": room_name,
                   "receiver_id": user_id},
                  check_error_callback)
