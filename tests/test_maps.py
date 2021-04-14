import unittest

from matplotlib import pyplot as plt

from avatar.mapworld.maps import ADEMap


class ADEMapTestCase(unittest.TestCase):

    def test_print_mapping(self):
        # Create map with five rooms on a four times three grid with no repetitions
        map = ADEMap(n=4, m=4, n_rooms=8, types_to_repeat=[2, 2])
        map.print_mapping()

    def test_plot_graph(self):
        map = ADEMap(n=4, m=3, n_rooms=10)
        map.print_mapping()
        map.plot_graph()
        plt.show()

    def test_to_json(self):
        """
        For example:
        {
        "directed": false,
        "multigraph": false,
        "graph": {},
        "nodes": [
            {
                "base_type": "outdoor",
                "type": "c/casino/outdoor",
                "target": false,
                "instance": "c/casino/outdoor/ADE_train_00005214.jpg",
                "id": [
                    3,
                    1
                ]
            },
            {
                "base_type": "indoor",
                "type": "h/hunting_lodge/indoor",
                "target": false,
                "instance": "h/hunting_lodge/indoor/ADE_train_00009734.jpg",
                "id": [
                    2,
                    1
                ]
            },
            ...
        ],
        "links": [
            {
                "source": [
                    3,
                    1
                ],
                "target": [
                    2,
                    1
                ]
            },
            ...
        ]
    }
        """
        map = ADEMap(n=4, m=3, n_rooms=5)
        map_json = map.to_json()
        print(map_json)


if __name__ == '__main__':
    unittest.main()
