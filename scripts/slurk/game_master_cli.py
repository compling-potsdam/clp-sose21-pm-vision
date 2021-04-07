"""
    Slurk client for the game master.

    Possibly requires:

    import sys
    sys.path.append("F:\\Development\\git\\clp-sose21-pm-vision")
"""

import click
import socketIO_client

from avatar.game_master_slurk import GameMaster


@click.command()
@click.option("--token", required=True)
@click.option("--slurk_host", default="127.0.0.1")
@click.option("--slurk_port", default="5000", type=int)
@click.option("--image_server_host", default="localhost")
@click.option("--image_server_port", default="8000", type=int)
def start_and_wait(token, slurk_host, slurk_port, image_server_host, image_server_port):
    custom_headers = {"Authorization": token, "Name": GameMaster.NAME}
    sio = socketIO_client.SocketIO(f'http://{slurk_host}', slurk_port, headers=custom_headers, Namespace=GameMaster)
    sio.get_namespace().set_base_image_url(f'http://{image_server_host}:{image_server_port}')
    sio.wait()


if __name__ == '__main__':
    start_and_wait()
