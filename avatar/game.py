import random
from avatar.mapworld.maps import ADEMap
from avatar.mapworld.mapworld import MapWorld

DIRECTION_TO_WORD = {
    "n": "north",
    "e": "east",
    "w": "west",
    "s": "south"
}


def direction_to_word(direction: str):
    if direction in DIRECTION_TO_WORD:
        return DIRECTION_TO_WORD[direction]
    return direction


def directions_to_sent(directions: str):
    if not directions:
        return "nowhere"
    n = len(directions)
    if n == 1:
        return direction_to_word(directions[0])
    words = [direction_to_word(d) for d in directions]
    return ", ".join(words[:-1]) + " or " + words[-1]


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

    def has_player(self, sid: int):
        return sid in self.get_players()

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
        self.done = False

    def is_ready(self):
        return self.has_player_with_role(MapWorldGame.ROLE_AVATAR) and len(self.get_players()) >= 2

    def has_started(self):
        return self.mapworld and self.target_state

    def set_done(self):
        self.done = True

    def is_done(self):
        return self.done

    def is_success(self):
        if self.has_started():
            return self.mapworld.state == self.target_state
        return False

    def is_avatar(self, sid: int):
        return self.get_game_role_for_player(sid) == MapWorldGame.ROLE_AVATAR

    def step(self, player: int, action: str):
        """
        Performs the action, when player is Avatar. Otherwise, we just return the current observation.

        :return: {
            "type": room_type,
            "instance": game_obs["descriptors"]["instance"],
            "situation": situation,
            "player": player,
            "directions": directions
        }
        """
        if not self.has_started():
            return None
        descriptors, directions = None, []
        # Guard mapworld transitions by player role
        if self.is_avatar(player):
            descriptors, directions = self.mapworld.try_transition(action)
        # Player is not avatar or avatar cannot move there
        if descriptors is None:
            return {"situation": "This is not possible. You can go %s." % (directions_to_sent(directions)),
                    "directions": directions,
                    "player": player,
                    "reward": -.1,  # small negative reward for each step
                    "done": False
                    }
        # Avatar movement was successful
        return self.get_observation(player)

    def reset(self, height: int, width: int, rooms: int, types_to_repeat: list):
        """
            Start random map.

        :param height: of the map
        :param width: of the map
        :param rooms: in the map
        """
        ademap = ADEMap(height, width, rooms, types_to_repeat=types_to_repeat)
        self.mapworld = MapWorld(ademap.to_fsa_def(), ['instance', 'type'])
        self.target_state = self.choose_random_target_state()
        self.done = False
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

    def get_mission(self, player: int):
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
        is_done = self.is_done()
        is_success = self.is_success()
        if is_done and is_success:
            reward = 1.
        elif is_done and not is_success:
            reward = -1.
        else:
            reward = -.1  # small negative reward for each step
        return {
            "type": room_type,
            "instance": game_obs["descriptors"]["instance"],
            "situation": "This is what you see. You can go %s." % (directions_to_sent(directions)),
            "player": player,
            "directions": directions,
            "reward": reward,
            "done": is_done
        }
