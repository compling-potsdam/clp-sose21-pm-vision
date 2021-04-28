"""
    Slurk client for the avatar
"""
import socketIO_client

from avatar.game_avatar_models import Avatar


def check_error_callback(success, error=None):
    if error:
        print("Error: ", error)


class AvatarBot(socketIO_client.BaseNamespace):
    """
        Listens to the users messages and execute move commands.

        The game rooms are already created with the setup script. We only join the rooms.

        The players might be already in the game room or join later.
    """
    ACTIONS = ["move", "respond"]
    NAME = "Avatar"

    def __init__(self, io, path):
        super().__init__(io, path)
        self.id = None
        self.model = None
        self.emit("ready")  # invokes on_joined_room for the token room so we can get the user.id

    def set_model(self, model: Avatar):
        self.model = model

    def on_joined_room(self, data: dict):
        """
        Send to the user itself to get the user_id.

        :param data: {'room': room.name, 'user': user.id}
        """
        print("on_joined_room", data)
        if not self.id:
            self.id = data['user']

    def on_observation(self, data: dict):
        """
            Receive the new observation for the bot.
        """
        obs = data["observation"]
        self.model.look_at(obs)

    def on_text_message(self, data: dict):
        """
            Receive and respond in a serial manner. (Would could also consider to create seperate response threads)

            :param data: {
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
        message = data["msg"]
        room_name = data["room"]
        user_name = data["user"]["name"]
        if user_name == "Game Master":
            return  # ignore game master messages for the bot; observations are sent via on_observation
        action = self.model.read_and_decide(message)
        if action not in AvatarBot.ACTIONS:
            print("Unknown avatar decision: " + action)
            return
        if action == "move":
            command = self.model.read_and_move(message)
            self.__send_command(command, room_name)
        if action == "respond":
            response = self.model.read_and_respond(message)
            self.__send_message(response, room_name)

    def __send_message(self, message, room_name):
        self.emit("text", {'room': room_name, 'msg': message}, check_error_callback)

    def __send_command(self, command, room_name):
        self.emit("message_command", {'room': room_name, 'command': command}, check_error_callback)
