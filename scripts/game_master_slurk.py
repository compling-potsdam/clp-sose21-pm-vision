"""
    Slurk client for the game master
"""
import click
import socketIO_client


def check_error_callback(success, error=None):
    if error:
        print("Error: ", error)


class GameMaster(socketIO_client.BaseNamespace):
    """
        Coordinates player between games.
    """

    def __init__(self, io, path):
        super().__init__(io, path)
        self.id = None
        self.emit("ready")  # invokes on_joined_room for the admin_room so we can get the user.id

    def on_status(self, data):
        """
        Send to room participants if a user joins or leaves the room.

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
        if user == self.id:
            print("Thats me...")
            return
        room_name = data["room"]
        self.emit("text", {"msg": f"Hello, {user}", "room": room_name}, check_error_callback)
        self.emit("image", {"url": "http://localhost:8000/training/a/abbey/ADE_train_00000970.jpg", "room": room_name},
                  check_error_callback)

    def __join_room(self, room_name):
        self.emit("join_room", {
            "user": self.id,
            "room": room_name
        }, check_error_callback)

    def on_new_room(self, data):
        print("on_new_room", data)
        self.__join_room(data["room"])

    def on_joined_room(self, data):
        """
        Send to the user itself.

        :param data: {'room': room.name, 'user': user.id}
        """
        print("on_joined_room", data)
        if self.id:
            return
        self.id = data['user']


@click.command()
@click.option("--token")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default="5000", type=int)
def start_and_wait(host, port, token):
    custom_headers = {"Authorization": token, "Name": "Game Master"}
    sio = socketIO_client.SocketIO(f'http://{host}', port, headers=custom_headers, Namespace=GameMaster)
    sio.wait()


if __name__ == '__main__':
    start_and_wait()
