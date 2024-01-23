from src import unpair, pair

class TestUnpair:
    def test_unpair_int(self):
        assert unpair(1) == 1

    def test_unpair_tuple(self):
        assert unpair((2, 2)) == 2


class TestPair:
    def test_pair_int(self):
        assert pair(1) == (1, 1)

    def test_pair_tuple(self):
        assert pair((2, 2)) == (2, 2)