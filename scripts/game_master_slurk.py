"""
    Slurk client for the game master
"""
import sys

sys.path.append("F:\\Development\\git\\clp-sose21-pm-vision")
print(sys.path)
import click
import socketIO_client

from avatar.game import MapWorldGame

GAME_MASTER = "Game Master"


def check_error_callback(success, error=None):
    if error:
        print("Error: ", error)


class GameMaster(socketIO_client.BaseNamespace):
    """
        Coordinates player between games.

        The game rooms are already created with the setup script. We only join the rooms.
    """

    def __init__(self, io, path):
        super().__init__(io, path)
        self.id = None
        self.base_image_url = None
        self.games = {}  # game by room_name
        self.emit("ready")  # invokes on_joined_room for the admin_room so we can get the user.id

    def set_base_image_url(self, base_image_url):
        self.base_image_url = base_image_url

    def on_joined_room(self, data):
        """
        Send to the user itself.

        :param data: {'room': room.name, 'user': user.id}
        """
        print("on_joined_room", data)
        if not self.id:
            self.id = data['user']
        room_name = data["room"]
        self.games[room_name] = MapWorldGame(room_name)

    def on_status(self, data):
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
        game = self.games[room_name]
        if data["type"] == "join":
            self.__send_private_message(f"Welcome, {user['name']}, I am your {GAME_MASTER}!", game.room, user["id"])
            self.__start_game_if_possible(user, game)
        if data["type"] == "leave":
            self.__pause_game(user, game)

    def __pause_game(self, user, game):
        ...

    def __start_game_if_possible(self, user, game):
        game.join(user["id"], user["name"])
        if game.is_ready():
            self.__start_game(game)
        else:
            self.__send_private_message("I will prepare the game for you now... this might take a while.!",
                                        game.room, user["id"])

    def __start_game(self, game: MapWorldGame):
        # Let player join the game room
        game.start_random_map(4, 4, 8)
        user_ids = game.get_players()
        for user_id in user_ids:
            self.__send_private_message("The game starts now... Have fun!", game.room, user_id)
            # Send initial observations
            observation = game.get_observation(user_id)
            self.__send_private_message(observation["situation"], game.room, user_id)
            self.__send_private_image(observation["instance"], game.room, user_id)
            # Send initial mission statements
            self.__send_private_message(game.get_mission(user_id), game.room, user_id)

    def __send_private_message(self, message, room_name, user_id):
        self.emit("text", {"msg": message, "room": room_name, "receiver_id": user_id}, check_error_callback)

    def __send_private_image(self, image_url, room_name, user_id):
        image_url = f"{self.base_image_url}/{image_url}"
        print(image_url)
        self.emit("image", {"url": image_url, "room": room_name, "receiver_id": user_id}, check_error_callback)


@click.command()
@click.option("--token")
@click.option("--slurk_host", default="127.0.0.1")
@click.option("--slurk_port", default="5000", type=int)
@click.option("--image_server_host", default="localhost")
@click.option("--image_server_port", default="8000", type=int)
def start_and_wait(token, slurk_host, slurk_port, image_server_host, image_server_port):
    custom_headers = {"Authorization": token, "Name": GAME_MASTER}
    sio = socketIO_client.SocketIO(f'http://{slurk_host}', slurk_port, headers=custom_headers, Namespace=GameMaster)
    sio.get_namespace().set_base_image_url(f'http://{image_server_host}:{image_server_port}')
    sio.wait()


if __name__ == '__main__':
    start_and_wait()
