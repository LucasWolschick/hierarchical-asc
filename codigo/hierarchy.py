hier = {
    "_root": ["_indoors", "_outdoors", "_transportation"],
    "_indoors": ["airport", "shopping_mall", "metro_station"],
    "_outdoors": [
        "street_pedestrian",
        "public_square",
        "street_traffic",
        "park",
    ],
    "_transportation": ["tram", "bus", "metro"],
}

paths = {}


def add_path(node, parents):
    paths[node] = parents + [node]
    if node in hier:
        for child in hier[node]:
            add_path(child, parents + [node])


add_path("_root", [])
