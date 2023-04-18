import cv2
import random 
import numpy as np
from dataclasses import dataclass
from game import sample, ranks

buffer = 10

@dataclass
class Screen:
    height: int
    width: int

class Target:
    def __init__(self, x, y, points, scale):
        self.points = points
        self.scale = scale
        self.x_off = x
        self.y_off = y
        self.x_end = x + scale
        self.y_end = y + scale
    
    def get_score(self):
        return self.points

def circular_mask(height, width):
    center = (int(width / 2), int(height / 2))
    radius = min(center[0], center[1], width - center[0], height - center[1])
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

# def get_offsets(height, width, group, buffer=10):
#     return (sample(width - group['scale'] - buffer, group['number'], group['scale']), 
#             np.array(np.random.rand(group['number']) * height, dtype=int))

# def get_offsets(height, width, group, buffer=10):
#     return (sample(width - group['scale'] - buffer, group['number'], group['scale']), 
#             np.full(group['number'], height - group['location'] - buffer, dtype=int))

def get_offsets(height, width, group, buffer=10):
    return (np.array(group['xs']), np.full(group['number'], height - group['ys'] - buffer, dtype=int))

def load_image(path, scale):
    return cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (scale, scale))

def mask_target(array, scale):
    return np.ma.array(array, mask=np.repeat(~circular_mask(scale, scale), 3))

def load_target(target):
    return mask_target(load_image('images/target.png', target.scale), target.scale)

def generate_targets(height, width, group):
    xs, ys = get_offsets(height, width, group, buffer)
    return [Target(xs[i], ys[i], group['points'], group['scale']) for i in range(group['number'])]

def generate_groups(dims, config):
    return [generate_targets(dims.height, dims.width, params) for group, params in config.items()]

def validate_targets(screen, targets, dims, config):
    try:
        _, _ = draw_targets(screen, targets)
        return targets
    except:
        return validate_targets(screen, generate_groups(dims, config), dims, config)
    
def load_coords(target):
    return target.y_off, target.y_end, target.x_off, target.x_end

def scale(action, dimensions):
    return (int(action[0]*dimensions.height), int(action[1]*dimensions.width))

def generate_target(target):
    y_off, y_end, x_off, x_end = load_coords(target)
    return load_target(target), y_off, y_end, x_off, x_end

def draw_target(screen, array, y_off, y_end, x_off, x_end):
    screen[y_off:y_end, x_off:x_end][~array.mask] = array[~array.mask]
    
def draw_mask(screen, array, y_off, y_end, x_off, x_end):
    screen[y_off:y_end, x_off:x_end][~array.mask] = np.ones(array[~array.mask].size)

def get_mask(target):
    xinds, yinds = np.where(~circular_mask(target.scale, target.scale) == False)
    return list(zip(yinds + target.y_off, xinds + target.x_off))

def get_count(targets):
    count = 0
    for group in targets:
        count += len(group)
    return count

def get_score(targets, position):
    for group in targets:
        for target in group:
            if position in get_mask(target):
                group.remove(target)
                return target.get_score()
    return -1

def get_reward(targets, position):
    score = get_score(targets, position)
    if get_count(targets) == 0:
        return score + 200, True
    return score, False

def draw_targets(screen, targets):
    background = screen.copy()
    mbackground = np.zeros_like(screen)
    for group in targets:
        for target in group:
            draw_target(background, *generate_target(target))
            draw_mask(mbackground, *generate_target(target))
    return background, mbackground[:, :, 0]

def draw_crosshair(screen, crosshair, location):
    # wow, what a mess - works though lol
    y = (crosshair.shape[0] // 2)
    x = (crosshair.shape[1] // 2)
    row = int(location[0] - y)
    col = int(location[1] - x)

    y_off = max(0, row)
    x_off = max(0, col)
    y_end = min(crosshair.shape[0] + row, screen.shape[0] - 1)
    x_end = min(crosshair.shape[1] + col, screen.shape[1] - 1)

    if (row < 0) & (col < 0):
        screen[y_off:y_end, x_off:x_end][~crosshair.mask[-row:, -col:, :]] = crosshair[-row:, -col:, :][~crosshair.mask[-row:, -col:, :]]
    elif ((crosshair.shape[0] + row) > (screen.shape[0] - 1)) & ((crosshair.shape[1] + col) > (screen.shape[1] - 1)):
        y_offset = -(crosshair.shape[0] + row - y_end)
        x_offset = -(crosshair.shape[1] + col - x_end)
        screen[y_off:y_end, x_off:x_end][~crosshair.mask[:y_offset, :x_offset, :]] = crosshair[:y_offset, :x_offset, :][~crosshair.mask[:y_offset, :x_offset, :]]
    elif (row < 0) & ((crosshair.shape[1] + col) > (screen.shape[1] - 1)):
        x_offset = -(crosshair.shape[1] + col - x_end)
        screen[y_off:y_end, x_off:x_end][~crosshair.mask[-row:, :x_offset, :]] = crosshair[-row:, :x_offset, :][~crosshair.mask[-row:, :x_offset, :]]
    elif ((crosshair.shape[0] + row) > (screen.shape[0] - 1))  & (col < 0):
        y_offset = -(crosshair.shape[0] + row - y_end)
        screen[y_off:y_end, x_off:x_end][~crosshair.mask[:y_offset, -col:, :]] = crosshair[:y_offset, -col:, :][~crosshair.mask[:y_offset, -col:, :]]
    elif (row < 0):
        screen[y_off:y_end, x_off:x_end][~crosshair.mask[-row:, :, :]] = crosshair[-row:, :, :][~crosshair.mask[-row:, :, :]]
    elif (col < 0):
        screen[y_off:y_end, x_off:x_end][~crosshair.mask[:, -col:, :]] = crosshair[:, -col:, :][~crosshair.mask[:, -col:, :]]
    elif (crosshair.shape[0] + row) > (screen.shape[0] - 1):
        y_offset = -(crosshair.shape[0] + row - y_end)
        screen[y_off:y_end, x_off:x_end][~crosshair.mask[:y_offset, :, :]] = crosshair[:y_offset, :, :][~crosshair.mask[:y_offset, :, :]]
    elif (crosshair.shape[1] + col) > (screen.shape[1] - 1):
        x_offset = -(crosshair.shape[1] + col - x_end)
        screen[y_off:y_end, x_off:x_end][~crosshair.mask[:, :x_offset, :]] = crosshair[:, :x_offset, :][~crosshair.mask[:, :x_offset, :]]
    else:
        screen[y_off:y_end, x_off:x_end][~crosshair.mask] = crosshair[~crosshair.mask]
        