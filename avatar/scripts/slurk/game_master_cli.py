"""
    Slurk client for the game master.

    Possibly requires:

    import sys
    sys.path.append("F:\\Development\\git\\clp-sose21-pm-vision")
"""
import base64

import click
import socketIO_client

from avatar.game_master_slurk import GameMaster


def build_url(host, context=None, port=None, base_url=None, auth=None):
    uri = "http://"
    if auth:
        uri = uri + f"{auth}@"  # seems not to work (try using query parameter)
    uri = uri + f"{host}"
    if context:
        uri = uri + f"/{context}"
    if port:
        uri = uri + f":{port}"
    if base_url:
        uri = uri + f"{base_url}"
    return uri


@click.command()
@click.option("--token", show_default=True, required=True,
              help="the master token for the game master bot. You get this afer game-setup. "
                   "The bot will join the token room.")
@click.option("--slurk_host", default="127.0.0.1", show_default=True, required=True,
              help="domain of the the slurk app")
@click.option("--slurk_context", default=None, show_default=True,
              help="sub-path of to the slurk host")
@click.option("--slurk_port", default="5000", show_default=True, required=True,
              help="port of the slurk app")
@click.option("--image_server_host", default="localhost", show_default=True, required=True,
              help="domain of the image server")
@click.option("--image_server_context", default=None, show_default=True,
              help="sub-path to the image server")
@click.option("--image_server_port", default="8000", show_default=True, required=True,
              help="port of the image server")
@click.option("--image_server_auth", help="credentials in format 'username:password'")
def start_and_wait(token, slurk_host, slurk_context, slurk_port,
                   image_server_host, image_server_context, image_server_port, image_server_auth):
    """Start the game master bot."""
    if slurk_port == "None":
        slurk_port = None

    if slurk_port:
        slurk_port = int(slurk_port)

    if image_server_port == "None":
        image_server_port = None

    if image_server_port:
        image_server_port = int(image_server_port)

    if image_server_auth == "None":
        image_server_auth = None

    custom_headers = {"Authorization": token, "Name": GameMaster.NAME}
    socket_url = build_url(slurk_host, slurk_context)
    sio = socketIO_client.SocketIO(socket_url, slurk_port, headers=custom_headers, Namespace=GameMaster)
    image_url = build_url(image_server_host, image_server_context, image_server_port)
    sio.get_namespace().set_base_image_url(image_url)
    if image_server_auth:
        # encode as base64, but keep as string
        image_server_auth = base64.b64encode(image_server_auth.encode("utf-8")).decode("utf-8")
        sio.get_namespace().set_image_server_auth(image_server_auth)
    sio.wait()


if __name__ == '__main__':
    start_and_wait()
