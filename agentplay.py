import cv2
import sys
import torch
import pygame
import imageio
import random
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from torchvision import transforms

from game import generate_groups, create_masks
from game import check_hit, get_count
from train import action_bounds, config, mparams
from train import screen_height, screen_width
from models import DActor, CActor


# writer = imageio.get_writer('cnngameplay2.gif', mode='I', fps=60)
np.random.seed(12)
@dataclass
class Action:
    x: Tuple[int, float]
    y: Tuple[int, float]

def scale(action, dimensions):
    return (int(action.x*dimensions.get_width()), int(action.y*dimensions.get_height()))

def get_action(models):
    if models == "cnn":
        return actor.select_greedy_action(transforms.Resize((128, 128))(torch.FloatTensor(pygame.surfarray.array3d(screen).T / 255.))).squeeze()
    elif models == "dnn":
        create_masks(mask_surface, targets)
        return actor.select_greedy_action(
            torch.FloatTensor(
                cv2.resize(pygame.surfarray.array3d(mask_surface).T[0, :, :] / 255., (128, 128)).reshape(32, 32, 4, 4).sum(axis=(2,3)).ravel() / 16.)).squeeze()


if __name__ == '__main__':
    levels = 2
    models = str(sys.argv[1])
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)

    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((screen_width, screen_height))
    mask_surface = pygame.Surface((screen_width, screen_height), flags=pygame.SRCALPHA)
    background_image = pygame.transform.scale(pygame.image.load('images/background.jpg').convert(), (screen_width, screen_height))
    crosshair_image = pygame.transform.scale(pygame.image.load('images/crosshair.png'), (int(.5*screen_width), int(.5*screen_width)))
    crosshair_rect = crosshair_image.get_rect()
    crosshair_rect.center = (screen_height // 2, screen_height // 2)
    pygame.mouse.set_visible(False)

    score = 0
    level = 1
    targets = generate_groups(screen, config)

    if models == "cnn":
        actor = CActor(action_bounds, mparams)
        actor.load_state_dict(torch.load('models/actor_000006.pt'))
    elif models == "dnn":
        actor = DActor(action_bounds, mparams)
        actor.load_state_dict(torch.load('models/actor_000008_520.pt'))
        # actor.load_state_dict(torch.load('actor_000009_520.pt'))

    ticks = 0
    running = True
    done = False
    clicks = 0
    action = np.array([0.5, 0.5])
    ilayerc = [black for _ in range(8)]
    hlayerc = [black for _ in range(7)]
    olayerc = [black for _ in range(2)]

    while running:
        if done:
            ilayerc = [black for _ in range(8)]
            hlayerc = [black for _ in range(7)]
            olayerc = [black for _ in range(2)]
        if ticks > 100 and ticks % 10 == 0:
            if not done:
                ilayerc = [random.choice([black, red]) for _ in range(8)]
                hlayerc = [random.choice([black, red]) for _ in range(7)]
                olayerc = [random.choice([black, red]) for _ in range(2)]
                action = get_action(models)
                print(action)
                clicks += 1
                for group in targets:
                    score = check_hit(group, scale(Action(action[1], action[0]), screen), score)
        # check for user events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(background_image, (0, 0))
        for group in targets:
            group.draw(screen)
        # draw_network(screen)
        input_layer = [pygame.Rect(20, y, 4, 4) for y in [45, 55, 65, 75, 85, 95, 105, 115]]
        hidden_layer = [pygame.Rect(60, y, 4, 4) for y in [50, 60, 70, 80, 90, 100, 110]]
        output_layer = [pygame.Rect(90, y, 4, 4) for y in [75, 85]]

        for il in input_layer:
            for hl in hidden_layer:
                pygame.draw.line(screen, white, (il.x + 4, il.y + 2), (hl.x, hl.y + 2), 1)
        for il in hidden_layer:
            for hl in output_layer:
                pygame.draw.line(screen, white, (il.x + 4, il.y + 2), (hl.x, hl.y + 2), 1)

        for i, r in enumerate(input_layer):
            pygame.draw.rect(screen, ilayerc[i], r)
        for i, r in enumerate(hidden_layer):
            pygame.draw.rect(screen, hlayerc[i], r)
        for i, r in enumerate(output_layer):
            pygame.draw.rect(screen, olayerc[i], r)

        crosshair_rect.center = scale(Action(action[1], action[0]), screen)
        screen.blit(crosshair_image, crosshair_rect)
        screen.blit(pygame.font.Font(None, 16).render("Score: " + str(score), True, (255, 255, 255)), (10, 10))
        screen.blit(pygame.font.Font(None, 16).render("Clicks: " + str(clicks), True, (255, 255, 255)), (75, 10))
        screen.blit(pygame.font.Font(None, 12).render(str(np.round(action[0], 3)), True, black), (96, 72))
        screen.blit(pygame.font.Font(None, 12).render(str(np.round(action[1], 3)), True, black), (96, 83))


        if (get_count(targets) == 0) and (level == levels):
            screen.blit(pygame.font.Font(None, 36).render("YOU WIN!", True, (255, 255, 255)), (.35*screen_height, .39*screen_width))
            done = True

        elif get_count(targets) == 0:
            targets = generate_groups(screen, config)
            level += 1
                

        pygame.display.update()
        # Append the image to the imageio writer
        # writer.append_data(pygame.surfarray.array3d(screen).transpose((1, 0, 2)))
        clock.tick(60)
        ticks += 1

    # Quit Pygame
    pygame.quit()
    # writer.close()