from logics.network import Socket
from logics.map import Tile
from logics import game_rules

PLAYER_CHARACTERS = ["A", "B", "C", "D"]
Trap_CHARACTERS = ["a", "b", "c", "d"]  # change_it


class Agent:
    def __init__(self, agent_id, tile, init_score, trap_count, connection: Socket):
        self.tile = tile
        self.init_score = init_score  # change_it maybe
        self._id = agent_id
        self._trap_count = trap_count  # change_it
        self.connection = connection
        self.trap_tiles = []
        self.gems = []
        self.hit_hurts = []
        self.trap_hurts = []
        self.turn_age = 0
        self.move_history = []
        self.keys = []

    @property
    def id(self):
        return self._id + 1

    @property
    def score(self):
        # change_it
        point = self.init_score
        gem_counts = self.get_gems_count()
        for i, gem_count in enumerate(gem_counts.values()):
            point += gem_count * game_rules.GEM_SCORES[i]
        point += len(self.hit_hurts) * game_rules.HIT_HURT
        point += len(self.trap_hurts) * game_rules.TRAP_HURT
        point += self.turn_age * game_rules.TURN_HURT
        return point

    @property
    def character(self):
        return PLAYER_CHARACTERS[self._id]

    @property
    def trap_character(self):  # change_it
        return Trap_CHARACTERS[self._id]

    @property
    def trap_count(self): # change_it
        return max([0, self._trap_count - len(self.trap_tiles)])

    def add_trap_tile(self, tile): # change_it
        self.trap_tiles.append(tile)

    @property
    def name(self):
        return PLAYER_CHARACTERS[self._id]

    def add_gem(self, gem):
        self.gems.append(gem)

    def add_key(self, key):
        # TODO validate
        # can agent agg duplicated keys??
        self.keys.append(key)

    def get_keys_count(self):
        return {
            "key1": self.gems.count(Tile.TileType.KEY1),
            "key2": self.gems.count(Tile.TileType.KEY2),
            "key3": self.gems.count(Tile.TileType.KEY3),
        }

    def has_key(self, key):
        return key in self.keys

    def get_gems_count(self):
        return {
            "gem1": self.gems.count(Tile.TileType.GEM1),
            "gem2": self.gems.count(Tile.TileType.GEM2),
            "gem3": self.gems.count(Tile.TileType.GEM3),
            "gem4": self.gems.count(Tile.TileType.GEM4),
        }

    def get_information(self):
        gem1, gem2, gem3, gem4 = self.get_gems_count().values()

        return {
            "score": self.score,
            "trap_count": self.trap_count,
            "hit_hurts_count": len(self.hit_hurts),
            "trap_hurts_count": len(self.trap_hurts),
            "gem1": gem1,
            "gem2": gem2,
            "gem3": gem3,
            "gem4": gem4,
        }
