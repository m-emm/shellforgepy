def get_bounding_box_center(bounding_box):
    min_point, max_point = bounding_box

    center_x = (min_point[0] + max_point[0]) / 2
    center_y = (min_point[1] + max_point[1]) / 2
    center_z = (min_point[2] + max_point[2]) / 2

    return (center_x, center_y, center_z)


def get_xlen(bounding_box):
    min_point, max_point = bounding_box
    return max_point[0] - min_point[0]


def get_ylen(bounding_box):
    min_point, max_point = bounding_box
    return max_point[1] - min_point[1]


def get_zlen(bounding_box):
    min_point, max_point = bounding_box
    return max_point[2] - min_point[2]


def get_xmin(bounding_box):
    min_point, _ = bounding_box
    return min_point[0]


def get_ymin(bounding_box):
    min_point, _ = bounding_box
    return min_point[1]


def get_zmin(bounding_box):
    min_point, _ = bounding_box
    return min_point[2]


def get_xmax(bounding_box):
    _, max_point = bounding_box
    return max_point[0]


def get_ymax(bounding_box):
    _, max_point = bounding_box
    return max_point[1]


def get_zmax(bounding_box):
    _, max_point = bounding_box
    return max_point[2]
