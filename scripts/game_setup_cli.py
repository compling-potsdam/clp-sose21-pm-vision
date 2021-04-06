"""
    Slurk client for the game master
"""
import click
import requests
import json
import socketIO_client


class SlurkApi:

    def __init__(self, bot_name, token, host="localhost", port="5000", base_url="/api/v2"):
        self.base_api = f"http://{host}:{port}{base_url}"
        self.get_headers = {"Authorization": f"Token {token}", "Name": bot_name}
        self.post_headers = {"Authorization": f"Token {token}",
                             "Name": bot_name,
                             "Content-Type": "application/json",
                             "Accept": "application/json"}

    def get_rooms(self):
        """
        :return: [
                    {'current_users': {},
                      'label': 'Admin Room',
                      'layout': 1,
                      'name': 'admin_room',
                      'read_only': False,
                      'show_latency': True,
                      'show_users': True,
                      'static': True,
                      'uri': '/room/admin_room',
                      'users': {'1': 'Game Master'}}
                  ]
        """
        r = requests.get(f"{self.base_api}/rooms", headers=self.get_headers)
        return json.loads(r.text)

    def create_room(self, payload):
        r = requests.post(f"{self.base_api}/room", headers=self.post_headers, data=json.dumps(payload))
        return json.loads(r.text)

    def get_tasks(self):
        """
        :return: [
                    {'current_users': {},
                      'label': 'Admin Room',
                      'layout': 1,
                      'name': 'admin_room',
                      'read_only': False,
                      'show_latency': True,
                      'show_users': True,
                      'static': True,
                      'uri': '/room/admin_room',
                      'users': {'1': 'Game Master'}}
                  ]
        """
        r = requests.get(f"{self.base_api}/tasks", headers=self.get_headers)
        return json.loads(r.text)

    def create_task(self, payload):
        r = requests.post(f"{self.base_api}/task", headers=self.post_headers, data=json.dumps(payload))
        return json.loads(r.text)

    def create_token(self, payload):
        r = requests.post(f"{self.base_api}/token", headers=self.post_headers, data=json.dumps(payload))
        return json.loads(r.text)


@click.command()
@click.option("--room_name", default="avatar_room")
@click.option("--task_name", default="avatar_game")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default="5000")
@click.option("--token", default="00000000-0000-0000-0000-000000000000")
def start_demo_game(room_name, task_name, host, port, token):
    slurk_api = SlurkApi("Game Setup", token, host, port)

    # Use the REST API to create a game task
    task_id = None
    task_exists = False
    tasks = slurk_api.get_tasks()
    for task in tasks:
        if task["name"] == task_name:
            task_exists = True
            task_id = task["id"]
            print(f"Task {task_name} already exists")
            break

    if not task_exists:
        print(f"Create game task '{task_name}'.")
        new_task = slurk_api.create_task({"name": task_name, "num_users": 3})
        task_id = new_task["id"]
        print(new_task)

    # Use the REST API to create a game room
    room_exists = False
    rooms = slurk_api.get_rooms()
    for room in rooms:
        if room["name"] == room_name:
            room_exists = True
            print("Game room already exists.")
            break

    if not room_exists:
        print(f"Create game room '{room_name}'.")
        print(slurk_api.create_room({
            "name": room_name,
            "label": "Avatar Game Room"
        }))

        # Use the Event API to publish room_created event
        sio = socketIO_client.SocketIO(f'http://{host}', port, headers={"Name": "Game Setup"})
        sio.emit("room_created", {"room": room_name, "task": task_id})
        sio.disconnect()

    # Use the REST API to create game tokens
    print("Master token: ", slurk_api.create_token({
        "room": room_name,
        "task": task_id,
        "message_text": True,
        "message_image": True,
        "user_room_join": True
    }))
    print("Player token: ", slurk_api.create_token({
        "room": room_name,
        "task": task_id,
        "message_text": True
    }))
    print("Avatar token: ", slurk_api.create_token({
        "room": room_name,
        "task": task_id,
        "message_text": True,
        "message_command": True
    }))


if __name__ == '__main__':
    start_demo_game()
