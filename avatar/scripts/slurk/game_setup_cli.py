"""
    Slurk client for the game master
"""
import click
import requests
import json
import socketIO_client

from avatar import load_project_resource


def build_url(host, context=None, port=None, base_url=None):
    uri = f"http://{host}"
    if context:
        uri = uri + f"/{context}"
    if port:
        uri = uri + f":{port}"
    if base_url:
        uri = uri + f"{base_url}"
    return uri


class SlurkApi:

    def __init__(self, bot_name, token, host="localhost", port="5000", context=None, base_url="/api/v2"):
        self.base_api = build_url(host, context, port, base_url)
        self.get_headers = {"Authorization": f"Token {token}", "Name": bot_name}
        self.post_headers = {"Authorization": f"Token {token}",
                             "Name": bot_name,
                             "Content-Type": "application/json",
                             "Accept": "application/json"}

    def get_layouts(self):
        """
        :return: [
                   {  'css': '',
                      'date_created': 1617785207,
                      'date_modified': 1617785207,
                      'html': '',
                      'id': 1,
                      'name': 'default',
                      'script': 'incoming_text = function(data) {...
                      'subtitle': '',
                      'title': '',
                      'uri': '/layout/1'}
                  ]
        """
        r = requests.get(f"{self.base_api}/layouts", headers=self.get_headers)
        return json.loads(r.text)

    def create_layout(self, payload):
        r = requests.post(f"{self.base_api}/layout", headers=self.post_headers, data=json.dumps(payload))
        return json.loads(r.text)

    def update_layout(self, payload, layout_id):
        r = requests.put(f"{self.base_api}/layout/%s" % layout_id, headers=self.post_headers, data=json.dumps(payload))
        return json.loads(r.text)

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

    def delete_room(self, room_name):
        r = requests.delete(f"{self.base_api}/room/{room_name}", headers=self.post_headers)
        return json.loads(r.text)

    def get_tasks(self):
        """
        :return: [
                   { 'date_created': 1617714526,
                      'date_modified': 1617714526,
                      'id': 1,
                      'layout': 1,
                      'name': 'avatar_game',
                      'num_users': 3,
                      'tokens': ['9b7f4039-e169-4016-bfa6-a9761b059391',
                       '24bdaf98-b9a8-43ab-9084-a9653ca5939d',
                       'fde65eac-a8b9-446b-a08c-be80125c7dbb'],
                      'uri': '/task/1'}
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
@click.option("--room_name", default="avatar_room", show_default=True, required=True)
@click.option("--task_name", default="avatar_game", show_default=True, required=True)
@click.option("--layout_name", default="avatar_layout", show_default=True, required=True)
@click.option("--slurk_host", default="127.0.0.1", show_default=True, required=True)
@click.option("--slurk_port", default="5000", show_default=True, required=True)
@click.option("--slurk_context", default=None, show_default=True)
@click.option("--token", default="00000000-0000-0000-0000-000000000000", show_default=True, required=True)
def setup_game(room_name, task_name, layout_name, slurk_host, slurk_port, slurk_context, token):
    """Setup the avatar game.

        \b
        TOKEN the admin token for the slurk rest api. You get this token, when starting slurk.
        ROOM_NAME the name of the slurk room to create
        TASK_NAME the name of the slurk task to create
        LAYOUT_NAME the name of the layout which will get uploaded for the room
        SLURK_HOST domain of the the slurk app
        SLURK_PORT port of the slurk app
        SLURK_CONTEXT (optional) sub-path to the slurk host
    """
    if slurk_port == "None":
        slurk_port = None

    if slurk_port:
        slurk_port = int(slurk_port)

    slurk_api = SlurkApi("Game Setup", token, slurk_host, slurk_port, slurk_context)

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

    # Use the REST API to create a game layout
    layout_id = None
    layout_exists = False
    layouts = slurk_api.get_layouts()
    for layout in layouts:
        if layout["name"] == layout_name:
            layout_exists = True
            layout_id = layout["id"]
            print(f"Layout {layout_name} already exists")
            break

    game_layout = load_project_resource("avatar/resources/game_layout.json")

    if not layout_exists:
        print(f"Create game layout '{layout_name}'.")
        new_layout = slurk_api.create_layout(game_layout)
        layout_id = new_layout["id"]

    # We update the layout. By default, the title is the name. But we want a different name.
    game_layout["name"] = layout_name
    print(f"Update layout {layout_name}")
    slurk_api.update_layout(game_layout, layout_id)

    # Use the REST API to create a game room
    rooms = slurk_api.get_rooms()
    for room in rooms:
        if room["name"] == room_name:
            print(f"Game room already exists. Removing old room with name '{room_name}'.")
            slurk_api.delete_room(room_name)
            break

    print(f"Create game room '{room_name}'.")
    print(slurk_api.create_room({
        "name": room_name,
        "label": "Avatar Game Room",
        "layout": layout_id
    }))

    # Use the Event API to publish room_created event
    socket_url = build_url(slurk_host, slurk_context)
    sio = socketIO_client.SocketIO(socket_url, slurk_port, headers={"Name": "Game Setup"})
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
        "message_text": True,
        "message_command": True
    }))
    print("Avatar token: ", slurk_api.create_token({
        "room": room_name,
        "task": task_id,
        "message_text": True,
        "message_command": True
    }))


if __name__ == '__main__':
    setup_game()
