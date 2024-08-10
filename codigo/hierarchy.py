# hier = {
#     "_root": ["_indoors", "_outdoors", "_transportation"],
#     "_indoors": ["airport", "shopping_mall", "metro_station"],
#     "_outdoors": [
#         "street_pedestrian",
#         "public_square",
#         "street_traffic",
#         "park",
#     ],
#     "_transportation": ["tram", "bus", "metro"],
# }

hier = {
    "_root": ["_1", "_2", "_3", "_4"],
    "_1": ["airport", "shopping_mall", "bus"],
    "_2": ["public_square", "park"],
    "_3": ["metro_station", "street_pedestrian", "street_traffic"],
    "_4": ["tram", "metro"],
}

paths = {}


def add_path(paths, hier, node, parents):
    paths[node] = parents + [node]
    if node in hier:
        for child in hier[node]:
            add_path(paths, hier, child, parents + [node])


add_path(paths, hier, "_root", [])
