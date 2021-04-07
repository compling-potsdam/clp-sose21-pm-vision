import random
from avatar.mapworld.maps import ADEMap
from avatar.mapworld.mapworld import MapWorld

DIRECTION_TO_WORD = {
    "n": "north",
    "e": "east",
    "w": "west",
    "s": "south"
}


class Game:
    """
        The state of a game.
    """
    COUNTER = 0

    def __init__(self, game_room: str = None, sid: int = None, game_role: str = None):
        self.players = dict()  # sid by game role / player name
        self.mapworld = None
        if game_room:
            self.room = game_room
        else:
            self.room = Game.COUNTER + 1
        if sid:
            self.join(sid, game_role)

    def join(self, sid: int, game_role: str):
        if game_role in self.players:
            raise Exception(f"Cannot join as {game_role} because already given.")
        self.players[game_role] = sid

    def get_player_by_game_role(self, game_role: str):
        return self.players[game_role]

    def get_game_role_for_player(self, sid: int):
        for role, player in self.players.items():
            if player == sid:
                return role
        return None

    def has_player_with_role(self, game_role: str):
        return game_role in self.players

    def get_players(self):
        return self.players.values()

    def get_game_room(self):
        return self.room


class MapWorldGame(Game):
    """
        The actual map worl environment. This is like the MapWorldWrapper.

        Multiple players can join, but there must be only one player with the 'Avatar' role!
    """
    ROLE_AVATAR = "Avatar"

    def __init__(self, game_room: str, sid: int = None, game_role: str = None):
        super().__init__(game_room, sid, game_role)
        self.mapworld = None
        self.target_state = None

    def is_ready(self):
        return self.has_player_with_role(MapWorldGame.ROLE_AVATAR) and len(self.get_players()) >= 2

    def has_started(self):
        return self.mapworld and self.target_state

    def is_done(self):
        if self.has_started():
            return self.mapworld.state == self.target_state
        return False

    def is_avatar(self, sid: int):
        return self.get_game_role_for_player(sid) == MapWorldGame.ROLE_AVATAR

    def reset(self, height: int, width: int, rooms: int):
        """
            Start random map.

        :param height: of the map
        :param width: of the map
        :param rooms: in the map
        """
        ademap = ADEMap(height, width, rooms)
        self.mapworld = MapWorld(ademap.to_fsa_def(), ['instance', 'type'])
        self.target_state = self.choose_random_target_state()
        return self.get_observations()

    def choose_random_target_state(self):
        # The target state should not be the initial avatar state
        avatar_state = self.mapworld.state
        other_states = [str(node["id"]) for node in self.mapworld.nodes if str(node["id"]) != avatar_state]
        assert avatar_state not in other_states
        return random.choice(other_states)

    def get_observations(self):
        return [self.get_observation(player) for player in self.get_players()]

    def __get_observation_internal(self, player: int):
        if self.is_avatar(player):
            descriptors, directions = self.mapworld.describe_node(self.mapworld.state)
        else:  # Director
            descriptors, directions = self.mapworld.describe_node(self.target_state)
        return {
            "player": player,
            "descriptors": descriptors,
            "directions": directions
        }

    def get_mission(self, player):
        mission = "This is your goal: "
        if self.is_avatar(player):
            mission += "Try to navigate to the director. The director will help you to lead you to his position."
        else:
            mission += "Try to navigate the avatar to your room. You can type anything that might help."
        return mission

    def get_observation(self, player: int):
        game_obs = self.__get_observation_internal(player)
        room_type = game_obs["descriptors"]["type"]
        if self.is_avatar(player):
            directions = game_obs["directions"]
        else:
            directions = []  # there is no movement for the director
        # Add situation statement
        situtation = "You see a %s and you can go %s." % (room_type.split("/")[1], self.directions_to_sent(directions))
        return {
            "type": room_type,
            "instance": game_obs["descriptors"]["instance"],
            "situation": situtation,
            "player": player,
            "directions": directions
        }

    def direction_to_word(self, direction: str):
        if direction in DIRECTION_TO_WORD:
            return DIRECTION_TO_WORD[direction]
        return direction

    def directions_to_sent(self, directions: str):
        if not directions:
            return "nowhere"
        n = len(directions)
        if n == 1:
            return self.direction_to_word(directions[0])
        words = [self.direction_to_word(d) for d in directions]
        return ", ".join(words[:-1]) + "or " + words[-1]
