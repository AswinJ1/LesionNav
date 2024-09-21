import pygame
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random

pygame.init()

window_width = 800
window_height = 600
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Grid with Brain")

image_folder = "Input Image"
image_files = sorted(os.listdir(image_folder))
output_folder = "Images With Labels"
os.makedirs(output_folder, exist_ok=True)

grid_background = Image.new("RGB", (window_width, window_height), (255, 255, 255))
grid_color = (0, 255, 0)
grid_spacing = 200
draw = ImageDraw.Draw(grid_background)

for x in range(0, window_width, grid_spacing):
    draw.line([(x, 0), (x, window_height)], fill=grid_color)

for y in range(0, window_height, grid_spacing):
    draw.line([(0, y), (window_width, y)], fill=grid_color)

class Agent:
    def __init__(self):
        self.agent_width = grid_spacing
        self.agent_height = grid_spacing
        self.agent_color = (255, 0, 0)
        self.agent_x = window_width // 2
        self.agent_y = window_height // 2
        self.DIRECTIONS = [(0, grid_spacing), (0, -grid_spacing), (grid_spacing, 0), (-grid_spacing, 0)]
        self.agent_speed = 1

    def move_agent(self):
        move_x, move_y = random.choice(self.DIRECTIONS)
        move_x *= self.agent_speed
        move_y *= self.agent_speed
        new_x = max(0, min(self.agent_x + move_x, window_width - self.agent_width))
        new_y = max(0, min(self.agent_y + move_y, window_height - self.agent_height))
        self.agent_x = new_x
        self.agent_y = new_y

agent = Agent()

image_index = 0
image_count = len(image_files)
reward = 0

clock = pygame.time.Clock()

reward = 0
episode = 0
reward_history = []
accuracy_history = []

def reset():
    global image_index, reward
    agent.agent_x = window_width // 2
    agent.agent_y = window_height // 2
    image_index = (image_index + 1) % image_count


def find_roi(image_path):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(grayscale_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_detected = len(contours) > 0
    return roi_detected

def apply_object_detection(image_path):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(grayscale_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, image)

def update_reward_plot(episode, reward_history, accuracy_history):
    print("Updating reward plot. Episode:", episode)
    print("Reward History:", reward_history)
    #plt.clf()

    # Plot Reward History
    plt.figure(1)
    plt.plot(range(1, episode + 1), reward_history)
    plt.title('Reward History')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # Plot Accuracy History
    plt.figure(2)
    plt.plot(range(1, episode + 1), accuracy_history)
    plt.title('Accuracy History')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')

    #plt.tight_layout()
    # Show plots
    plt.show(block=False)
    plt.pause(0.1)

num_episodes=300
running = True

while running and episode < num_episodes:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    image_path = os.path.join(image_folder, image_files[image_index])
    brain_image = pygame.image.load(image_path)
    brain_image = pygame.transform.scale(brain_image, (window_width, window_height))
    screen.blit(brain_image, (0, 0))

    for x in range(0, window_width, grid_spacing):
        pygame.draw.line(screen, grid_color, (x, 0), (x, window_height))

    for y in range(0, window_height, grid_spacing):
        pygame.draw.line(screen, grid_color, (0, y), (window_width, y))

    agent.move_agent()
    pygame.draw.rect(screen, agent.agent_color, (agent.agent_x, agent.agent_y, agent.agent_width, agent.agent_height))

    pygame.display.flip()

    roi_detected = find_roi(image_path)

    if roi_detected:
        print("Lesion detected in image:", image_files[image_index])
        apply_object_detection(image_path)
        reward += 2
        accuracy =1
        print("Reward:", reward)
        print("Accuracy:",accuracy)
        reset()
        episode += 1
        reward_history.append(reward)

        # Dummy accuracy value, replace it with your actual accuracy calculation
        accuracy = random.uniform(0, 1)
        accuracy_history.append(accuracy)
        update_reward_plot(episode, reward_history, accuracy_history)

    else:
        reward -= 1
        accuracy=0
        print("Reward:", reward)
        print("Accuracy:",accuracy)
        reset()
        episode+=1
        reward_history.append(reward)
        
        
        # Dummy accuracy value, replace it with your actual accuracy calculation
        #accuracy = random.uniform(0, 1)
        accuracy_history.append(accuracy)
        update_reward_plot(episode, reward_history, accuracy_history)

    clock.tick(1)

pygame.quit()