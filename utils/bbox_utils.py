def get_bbox_width(bbox):
    """
    Calculate the width of a bounding box.

    Parameters:
    bbox (list or tuple): A list or tuple containing the coordinates of the bounding box in the format [x1, y1, x2, y2].

    Returns:
    int: The width of the bounding box.
    """
    x1, _, x2, _ = bbox
    return int(x2 - x1)

def measure_distance(p1,p2):
    return((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return int(center_x), int(center_y)


def get_foot_position(bbox):
    """Return the bottom-center of a player box (the point touching the court)."""
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

    
