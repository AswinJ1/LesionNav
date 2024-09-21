import pygame
import numpy as np
import os
import cv2

class BrainEnv:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
        self.current_image_index = 0
        self.grid_size = 60
        self.agent_position = (0, 0)
        self.screen = pygame.display.set_mode((240, 240))
        self.font = pygame.font.SysFont(None, 24)
        self.current_image = None

    def reset(self):
        self.agent_position = (0, 0)
        while True:
            self.current_image = cv2.imread(self.images[self.current_image_index], cv2.IMREAD_GRAYSCALE)
            self.current_image_index = (self.current_image_index + 1) % len(self.images)
            if np.any(self.current_image):  # Skip blank images
                break
        return self._get_state()

    def step(self, action):
        x, y = self.agent_position
        if action == 0:  # stay
            pass
        elif action == 1:  # move down
            y = min(y + self.grid_size, 240 - self.grid_size)
        elif action == 2:  # move right
            x = min(x + self.grid_size, 240 - self.grid_size)
        elif action == 3:  # move up
            y = max(y - self.grid_size, 0)
        elif action == 4:  # move left
            x = max(x - self.grid_size, 0)

        self.agent_position = (x, y)
        reward = self._get_reward()
        done = self._check_done()
        self.render()

        return self._get_state(), reward, done, {}

    def render(self):
        self.screen.fill((0, 0, 0))
        img_surface = pygame.surfarray.make_surface(cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB))
        self.screen.blit(img_surface, (0, 0))

        # Draw grid
        for x in range(0, 240, self.grid_size):
            for y in range(0, 240, self.grid_size):
                rect = pygame.Rect(x, y, self.grid_size, self.grid_size)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

        # Draw agent
        agent_rect = pygame.Rect(self.agent_position[0], self.agent_position[1], self.grid_size, self.grid_size)
        pygame.draw.rect(self.screen, (255, 0, 0), agent_rect, 2)

        pygame.display.flip()

    def _get_state(self):
        x, y = self.agent_position
        state = self.current_image[y:y+self.grid_size, x:x+self.grid_size]
        state = cv2.resize(state, (60, 60))
        return state

    def _get_reward(self):
        x, y = self.agent_position
        region = self.current_image[y:y+self.grid_size, x:x+self.grid_size]
        if np.any(region == 255):  # Assuming tumor regions are white
            return 5  # Positive reward for finding the tumor (previous value was 5)
        return -0.01  # Small penalty for moving

    def _check_done(self):
        x, y = self.agent_position
        region = self.current_image[y:y+self.grid_size, x:x+self.grid_size]
        return np.any(region == 255)

    def save_image_with_label(self, filename):
        x, y = self.agent_position
        img_with_label = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(img_with_label, (x, y), (x + self.grid_size, y + self.grid_size), (0, 255, 0), 2)
        cv2.imwrite(filename, img_with_label)
