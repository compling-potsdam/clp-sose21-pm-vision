import random
from avatar.mapworld.maps import ADEMap
from avatar.mapworld.mapworld import MapWorld


class Game:
    """
        The state of a game.
    """
    COUNTER = 0

    def __init__(self, sid, game_role):
        self.game_id = Game.COUNTER + 1
        self.players = dict()  # sid by game role
        self.join(sid, game_role)
        self.mapworld = None

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


class MapWorldGame(Game):
    """
        The actual map worl environment. This is like the MapWorldWrapper.
    """

    def __init__(self, sid, game_role):
        super().__init__(sid, game_role)
        self.mapworld = None
        self.target_node = None

    def start_random_map(self, height, width, rooms):
        ademap = ADEMap(height, width, rooms)
        self.mapworld = MapWorld(ademap.to_fsa_def(), ['instance', 'type'])
        self.target_node = self.choose_random_target_node()
        return self.get_observations()

    def choose_random_target_node(self):
        # The target state should not be the initial avatar state
        avatar_state = self.mapworld.state
        other_states = [str(node["id"]) for node in self.mapworld.nodes if str(node["id"]) != avatar_state]
        assert avatar_state not in other_states
        return random.choice(other_states)

    def get_observations(self):
        return [self.get_observation(player) for player in self.get_players()]

    def get_observation(self, player):
        game_role = self.get_game_role_for_player(player)
        if game_role not in ["Avatar", "Director"]:
            print(f"Unknown game role: {game_role}")
            return None
        if game_role == "Avatar":
            descriptors, directions = self.mapworld.describe_node(self.mapworld.state)
        else:  # Director
            descriptors, directions = self.mapworld.describe_node(self.target_node)
        return {
            "player": player,
            "role": game_role,
            "descriptors": descriptors,
            "directions": directions
        }
