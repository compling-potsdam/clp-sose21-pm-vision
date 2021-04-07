"""
    Standalone server for the game master
"""
import socketio

from avatar.game import MapWorldGame

REQ_HEADERS = ["X-Game-Mode", "X-Role"]
MSG_WELCOME = "Welcome, I am your Game Master!"
MSG_AWAITING = "I will prepare the game for you now... this might take a while."
MSG_GAME_START = "The game starts now... Have fun!"
GAME_MASTER = "Game Master"


def has_required_headers(headers):
    for h in REQ_HEADERS:
        if h not in headers:
            return False
    return True


def get_game_mode(headers):
    return headers["X-Game-Mode"]


def get_game_role(headers):
    return headers["X-Role"]


class GameMaster(socketio.Namespace):
    """
        Coordinates player between games.

        We create game rooms on-the-fly.
    """

    def __init__(self):
        super(GameMaster, self).__init__()
        self.games = {}  # game by sid (the same game is referenced twice)

    def get_awaiting_demo_game_missing(self, game_role):
        for sid, game in self.games.items():
            if not game.has_player_with_role(game_role):
                return game
        return None

    def start_demo_game(self, sid, game_role):
        self.send({"from": GAME_MASTER, "msg": MSG_WELCOME}, room=sid)
        # We check for games to join
        game = self.get_awaiting_demo_game_missing(game_role)
        if game:
            # We join an awaiting game ...
            game.join(sid, game_role)
            # ... register the player ...
            self.games[sid] = game
            # ... and start the game
            self.start_game(game)
        else:
            # We create a new awaiting game, so that another user can connect later as either director or avatar.
            self.games[sid] = MapWorldGame(sid, game_role)  # there should be at least one player per game
            # We wait until someone elses joins
            self.send({"from": GAME_MASTER, "msg": MSG_AWAITING}, room=sid)

    def start_standard_game(self, sid):
        # We start right away. An automatic agent is created for the users interaction.
        self.send({"from": GAME_MASTER, "msg": MSG_GAME_START}, room=sid)
        ...

    def start_game(self, game: MapWorldGame):
        # Let player join the game room
        sids = game.get_players()
        game_room = game.get_game_room()
        for sid in sids:
            self.enter_room(sid, game_room)
        self.send({"from": GAME_MASTER, "msg": MSG_GAME_START}, room=game_room)
        # Send initial observations
        initial_observations = game.reset(4, 4, 8)
        for initial_observation in initial_observations:
            self.send_observation(initial_observation)
        # Send initial mission statements
        for sid in sids:
            game_role = game.get_game_role_for_player(sid)
            mission = "This is your goal: "
            if game_role == "Director":
                mission += "Try to navigate the avatar to your room. You can type anything that might help."
            else:
                mission += "Try to navigate to the director. The director will help you to lead you to his position."
            self.send({"from": GAME_MASTER, "msg": mission}, room=sid)

    def send_observation(self, game_obs):
        data = dict()
        data["from"] = GAME_MASTER
        data["observation"] = game_obs
        # Add situation statement
        message = "This is your situation: %s" % game_obs["situation"]
        data["msg"] = message
        self.emit("observation", data, room=game_obs["player"])

    def on_connect(self, sid, env, auth):
        connection_headers = dict(env["headers_raw"])  # headers_raw is a tuple of tuples
        if not has_required_headers(connection_headers):
            self.send({"from": GAME_MASTER, "msg": "Error: Missing one or more required headers: %s" % REQ_HEADERS},
                      room=sid)
            self.disconnect(sid)
        game_mode = get_game_mode(connection_headers)
        if game_mode == "demo":
            # For the demo the players can join as a given role
            game_role = get_game_role(connection_headers)
            self.start_demo_game(sid, game_role)
        elif game_mode == "standard":
            # We dont need the role here, because the player is always the director
            self.start_standard_game(sid)
        else:
            # No valid game_mode
            self.disconnect(sid)

    def on_disconnect(self, sid):
        if sid not in self.games:
            # Disconnected before game established
            pass
        else:
            # We remove the entries from the games registry
            game = self.games[sid]
            sids = game.get_players()
            for sid in sids:
                self.send({"from": GAME_MASTER, "msg": "Game ended, because a player disconnected."}, room=sid)
                del self.games[sid]

    def on_command(self, sid, data):
        # TODO special handling of avatar commands (changing the game state)
        ...

    def on_message(self, sid, data):
        if sid not in self.games:
            pass
        # We generalize this by re-sending into the same game room and skipping the sender (this might allow spectators)
        game = self.games[sid]
        game_room = game.get_game_room()
        # We prefix the message with the players game role
        game_role = game.get_game_role_for_player(sid)
        self.send({"from": game_role, "msg": data}, room=game_room, skip_sid=sid)
