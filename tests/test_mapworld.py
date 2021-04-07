import unittest

from avatar.game import MapWorldGame


class MapWorldGameTestCase(unittest.TestCase):

    def test_start_random_map(self):
        game = MapWorldGame("player_1", "Director")
        init_obs = game.reset(4, 4, 5)
        print(init_obs)


if __name__ == '__main__':
    unittest.main()
