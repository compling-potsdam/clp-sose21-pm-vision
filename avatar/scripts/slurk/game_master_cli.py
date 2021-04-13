"""
    Slurk client for the game master.

    Possibly requires:

    import sys
    sys.path.append("F:\\Development\\git\\clp-sose21-pm-vision")
"""

import click
import socketIO_client

from avatar.game_master_slurk import GameMaster


def build_url(host, context=None, port=None, base_url=None):
    uri = f"http://{host}"
    if context:
        uri = uri + f"/{context}"
    if port:
        uri = uri + f":{port}"
    if base_url:
        uri = uri + f"{base_url}"
    return uri


@click.command()
@click.option("--token", show_default=True, required=True)
@click.option("--slurk_host", default="127.0.0.1", show_default=True, required=True)
@click.option("--slurk_context", default=None, show_default=True)
@click.option("--slurk_port", default="5000", show_default=True, required=True)
@click.option("--image_server_host", default="localhost", show_default=True, required=True)
@click.option("--image_server_context", default=None, show_default=True)
@click.option("--image_server_port", default="8000", show_default=True, required=True)
def start_and_wait(token, slurk_host, slurk_context, slurk_port,
                   image_server_host, image_server_context, image_server_port):
    """Start the game master bot.

        \b
        TOKEN the master token for the game master bot. You get this afer game-setup. The bot will join the token room.
        SLURK_HOST domain of the the slurk app
        SLURK_PORT port of the slurk app
        SLURK_CONTEXT (optional) sub-path of to the slurk host
        IMAGE_SERVER_HOST domain of the image server
        IMAGE_SERVER_PORT port of the image server
        IMAGE_SERVER_CONTEXT (optional) sub-path to the image server
    """
    if slurk_port == "None":
        slurk_port = None

    if slurk_port:
        slurk_port = int(slurk_port)

    if image_server_port == "None":
        image_server_port = None

    if image_server_port:
        image_server_port = int(image_server_port)

    custom_headers = {"Authorization": token, "Name": GameMaster.NAME}
    socket_url = build_url(slurk_host, slurk_context)
    sio = socketIO_client.SocketIO(socket_url, slurk_port, headers=custom_headers, Namespace=GameMaster)
    image_url = build_url(image_server_host, image_server_context, image_server_port)
    sio.get_namespace().set_base_image_url(image_url)
    sio.wait()


if __name__ == '__main__':
    start_and_wait()
