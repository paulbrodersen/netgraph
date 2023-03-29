single_edge = [
    (0, 1),
]

satellites = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7)
]

cycle = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 0)
]

triangle = [(0, 1), (1, 2), (2, 0)]

cube = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7)
]

star = [(0, ii) for ii in range(1, 20)]

unbalanced_tree = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (2, 6),
    (3, 7),
    (3, 8),
    (4, 9),
    (4, 10),
    (4, 11),
    (5, 12),
    (5, 13),
    (5, 14),
    (5, 15)
]

balanced_tree = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 7),
    (2, 8),
    (2, 9),
    (3, 10),
    (3, 11),
    (3, 12),
    (4, 13),
    (4, 14),
    (4, 15),
    (5, 16),
    (5, 17),
    (5, 18),
    (6, 19),
    (6, 20),
    (6, 21),
    (7, 22),
    (7, 23),
    (7, 24),
    (8, 25),
    (8, 26),
    (8, 27),
    (9, 28),
    (9, 29),
    (9, 30),
    (10, 31),
    (10, 32),
    (10, 33),
    (11, 34),
    (11, 35),
    (11, 36),
    (12, 37),
    (12, 38),
    (12, 39),
]

chain = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
]

bipartite = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
]
