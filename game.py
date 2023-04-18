import pygame
import random 
import numpy as np

buffer = 5
screen_width, screen_height = 1000, 600

config = {

    "large": {
        "number": 5,
        "location": int(.25*screen_height),
        "scale": int(.15*screen_width),
        "points": 10
        },
    "medium": {
        "number": 5,
        "location": int(.35*screen_height),
        "scale": int(.07*screen_width),
        "points": 25
        },
    "small": {
        "number": 5,
        "location": int(.45*screen_height),
        "scale": int(.05*screen_width),
        "points": 30
        }

}

class Target(pygame.sprite.Sprite):
    def __init__(self, image, x, y, points):
        super().__init__()
        self.image = image
        self.points = points
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
    
    def get_score(self):
        return self.points
    
def generate_targets(screen, group):
    xs, ys = get_offsets(screen, group, buffer)
    return pygame.sprite.Group(
        [Target(load_image('images/target.png', group['scale']), xs[i], ys[i], group['points']) for i in range(group['number'])]
    )

def ranks(sample):
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])

def sample(n=40, k=4, d=10):
    sample = random.sample(range(n-(k-1)*(d-1)), k)
    return np.array([s + (d-1)*r for s, r in zip(sample, ranks(sample))])

def get_offsets(screen, group, buffer=10):
    try:
        return (sample(screen.get_width() - group['scale'] - buffer, group['number'], group['scale']) + buffer, 
            np.full(group['number'], screen.get_height() - group['location'] - buffer))
    except:
        return (np.array(group['xs']), np.full(group['number'], screen.get_height() - group['ys'] - buffer, dtype=int))

def load_image(path, scale):
    return pygame.transform.scale(pygame.image.load(path).convert_alpha(), (scale, scale))

def generate_groups(screen, config):
    return [generate_targets(screen, params) for group, params in config.items()]

def check_hit(target_group, position, score):
    for target in target_group:
        if target.rect.collidepoint(position[0], position[1]):
            score += target.get_score()
            target_group.remove(target)
    return score

def get_count(targets):
    count = 0
    for group in targets:
        count += len(group.sprites())
    return count

def create_masks(mask_surface, targets):
    mask_surface.fill((0, 0, 0, 255))
    sprite_masks = [(pygame.surfarray.pixels_alpha(target.image), (target.rect.left, target.rect.top)) for group in targets for target in group]
    sprite_masks = [(np.where(mask[0] > 0, 255, 0), mask[1]) for mask in sprite_masks]
    sprite_masks = [(pygame.surfarray.make_surface(mask[0]), mask[1]) for mask in sprite_masks]

    # blit the sprite masks onto the mask surface
    for sprite_mask in sprite_masks:
        mask_surface.blit(sprite_mask[0], sprite_mask[0].get_rect(topleft=(sprite_mask[1])))

if __name__ == '__main__':
    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((screen_width, screen_height))
    background_image = pygame.transform.scale(pygame.image.load('images/background.jpg').convert(), (screen_width, screen_height))
    crosshair_image = pygame.transform.scale(pygame.image.load('images/crosshair.png'), (int(.3*screen_width), int(.3*screen_width)))
    crosshair_rect = crosshair_image.get_rect()
    pygame.mouse.set_visible(False)

    score = 0
    targets = generate_groups(screen, config)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for group in targets:
                        score = check_hit(group, event.pos, score)

        screen.blit(background_image, (0, 0))
        for group in targets:
            group.draw(screen)

        crosshair_rect.center = pygame.mouse.get_pos()
        screen.blit(crosshair_image, crosshair_rect)
        screen.blit(pygame.font.Font(None, 36).render("Score: " + str(score), True, (255, 255, 255)), (10, 10))

        if get_count(targets) == 0:
            screen.blit(pygame.font.Font(None, 36).render("YOU WIN!", True, (255, 255, 255)), (.4*screen_height, 0.4*screen_width))

        pygame.display.update()
        clock.tick(60)

    # Quit Pygame
    pygame.quit()


