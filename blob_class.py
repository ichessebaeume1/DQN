import numpy as np
import cv2
from PIL import Image


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        """
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        """
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x, y):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size - 1:
            self.x = self.size - 1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size - 1:
            self.y = self.size - 1


class BlobEnv:
    size = 10
    return_images = True
    move_penalty = 1
    enemy_penalty = 300
    food_reward = 25
    observation_space_values = (size, size, 3)  # 4
    action_space_size = 9
    player_n = 1  # player key in dict
    food_n = 2  # food key in dict
    enemy_n = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.size)
        self.food = Blob(self.size)
        while self.food == self.player:
            self.food = Blob(self.size)
        self.enemy = Blob(self.size)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.size)

        self.episode_step = 0

        if self.return_images:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food) + (self.player - self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        # self.enemy.move()
        # self.food.move()
        ##############

        if self.return_images:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player - self.food) + (self.player - self.enemy)

        if self.player == self.enemy:
            reward = -self.enemy_penalty
        elif self.player == self.food:
            reward = self.food_reward
        else:
            reward = -self.move_penalty

        done = False
        if reward == self.food_reward or reward == -self.enemy_penalty or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.size, self.size, 3), dtype=np.uint8)  # starts a rbg of our size
        env[self.food.x][self.food.y] = self.d[self.food_n]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.enemy_n]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.player_n]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img
