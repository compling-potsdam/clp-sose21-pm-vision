import socketio

REQ_HEADERS = ["X-Game-Mode", "X-Role"]
MSG_WELCOME = "Game Master: Welcome, I am your Game Master!"
MSG_AWAITING = "Game Master: I will prepare the game for you now... this might take a while."
MSG_GAME_START = "Game Master: The game starts now... Have fun!"


def has_required_headers(headers):
    for h in REQ_HEADERS:
        if h not in headers:
            return False
    return True


def get_game_mode(headers):
    return headers["X-Game-Mode"]


def get_game_role(headers):
    return headers["X-Role"]


class Game:
    """
        The state of a game.
    """
    COUNTER = 0

    def __init__(self, sid, game_role):
        self.game_id = Game.COUNTER + 1
        self.players = dict()  # sid by game role
        self.join(sid, game_role)

    def join(self, sid, game_role):
        if game_role in self.players:
            raise Exception(f"Cannot join as {game_role} because already given.")
        self.players[game_role] = sid

    def get_player_by_game_role(self, game_role):
        return self.players[game_role]

    def get_game_role_for_player(self, sid):
        for role, player in self.players.items():
            if player == sid:
                return role
        return None

    def has_player_with_role(self, game_role):
        return game_role in self.players

    def get_players(self):
        return self.players.values()

    def get_game_room(self):
        return self.game_id


class GameMaster(socketio.Namespace):
    """
        Coordinates player between games.

        Maybe have one game master per game?
        But then we have to match sid to game masters. Which might have the same effort.
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
        self.send(MSG_WELCOME, room=sid)
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
            self.games[sid] = Game(sid, game_role)  # there should be at least one player per game
            # We wait until someone elses joins
            self.send(MSG_AWAITING, room=sid)

    def start_standard_game(self, sid):
        # We start right away. An automatic agent is created for the users interaction.
        self.send(MSG_GAME_START, room=sid)
        ...

    def start_game(self, game: Game):
        # Let player join the game room
        sids = game.get_players()
        game_room = game.get_game_room()
        for sid in sids:
            self.enter_room(sid, game_room)
        self.send(MSG_GAME_START, room=game_room)
        # TODO send initial stuff to players
        ...

    def on_connect(self, sid, env, auth):
        connection_headers = dict(env["headers_raw"])  # headers_raw is a tuple of tuples
        if not has_required_headers(connection_headers):
            self.send("Error: Missing one or more required headers: %s" % REQ_HEADERS, room=sid)
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
                self.send("Game Master: Game ended, because a player disconnected.", room=sid)
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
        self.send(f"{game_role}: {data}", room=game_room, skip_sid=sid)
