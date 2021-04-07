"""
    Standalone server for the game master
"""
import click
import eventlet
import socketio

from avatar.game_master_standalone import GameMaster


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default="5555", type=int)
def start_and_wait(host, port):
    sio = socketio.Server()
    app = socketio.WSGIApp(sio)

    sio.register_namespace(GameMaster())  # default namespace handler
    eventlet.wsgi.server(eventlet.listen((host, port)), app)


if __name__ == '__main__':
    start_and_wait()
