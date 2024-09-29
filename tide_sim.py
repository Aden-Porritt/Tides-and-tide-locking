import numpy as np
import math
import numba
import tide_sim_eng as eng

def dot(vector1, vector2):
    return vector1[0] * vector2[0] + vector1[1] * vector2[1]

def norm(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)

class Body:
    def __init__(self):
        self.nodes_pos = []
        self.nodes_velocity = []

        self.springs_nodes = []
        self.springs_s = []
        self.start_springs_length = np.array([])
        self.last_muscles_length = np.array([])

        self.node_size = 20
        self.r = 200
        self.data = []
        self.x = []
        self.last_x = 0
        self.angle_data = [0]

    def show_sim(self):
        self.nodes_pos = np.array(self.nodes_pos, 'f8')
        self.nodes_velocity = np.array(self.nodes_velocity, 'f8')

        self.springs_nodes = np.array(self.springs_nodes, 'i4')
        self.springs_s = np.array(self.springs_s, 'f8')

        fps = 2**2
        clock = pygame.time.Clock()
        self.start_springs_length, vector = eng.get_springs_length(self.nodes_pos, self.springs_nodes)
        self.nodes_pos = self.nodes_pos.copy()
        self.nodes_velocity = np.zeros((len(self.nodes_pos), 2), 'f8')
        for i in range(1, len(self.nodes_velocity)):
            print(i)
            self.nodes_velocity[i] = np.array([self.nodes_pos[i][1], -self.nodes_pos[i][0]]) / 10
            print(self.nodes_velocity[i])
        self.last_springs_length = self.start_springs_length.copy()
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        fps = math.ceil(fps / 2)
                        print('kl')
                    if event.key == pygame.K_RIGHT:
                        fps *= 2
                        print('kr')
                    if event.key == pygame.K_0:
                        return True
            self.nodes_pos, self.nodes_velocity, self.last_springs_length = eng.move(self.nodes_pos, self.nodes_velocity, self.start_springs_length, self.springs_nodes, self.springs_s, self.last_springs_length, 0.01, fps)
            self.draw_animal()
            # self.data.append(self.nodes_pos[1] - eng.get_center(self.nodes_pos))
            avg_node_v = np.sum(self.nodes_velocity, axis=0) / len(self.nodes_velocity)
            self.nodes_velocity = self.nodes_velocity - avg_node_v
            q = 0.0
            cen = eng.get_center(self.nodes_pos)
            self.nodes_pos -= cen
            for index, node_v in enumerate(self.nodes_velocity):
                q += -(np.dot(np.array([node_v[1], -node_v[0]]), (self.nodes_pos[index])) * np.sqrt(np.sum((self.nodes_pos[index]) ** 2)))
            self.data.append(q / len(self.nodes_velocity))
            self.x.append(self.last_x + fps)
            self.last_x += fps
            if len(self.data) > 1:
                self.angle_data.append((self.data[-2] - self.data[-1]) / fps)
            print(fps, self.data[-1], self.angle_data[-1])
        return True

    def draw_animal(self):
        self.center = eng.get_center(self.nodes_pos)
        WIN.fill((1 , 1, 1))
        self.draw_muscles()
        self.draw_nodes()
        pygame.draw.circle(WIN, (0, 0, 255), [WIDTH / 2, HEIGHT / 2], self.r + 2, 5)
        pygame.display.update()
        

    def draw_muscles(self):
        for i, nodes in enumerate(self.springs_nodes):
            pos1, pos2 = self.nodes_pos[nodes[0]], self.nodes_pos[nodes[1]]
            draw_rectangle(pos1, pos2, 10, self.springs_s[i], self.center, self.node_size)

    def draw_nodes(self):
        colour = [(255, 255, 255) for _ in self.nodes_pos]
        for i, node in enumerate(self.nodes_pos):
            pos = [WIDTH / 2 + node[0] - self.center[0], HEIGHT / 2 + self.center[1] - node[1]]
            pygame.draw.circle(WIN, colour[i], pos, self.node_size, 0)

def draw_rectangle(pos1, pos2, width, colour, cen, size):
    leght = norm(pos1 - pos2)
    if pos1[1] >= pos2[1]:
        up = pos1
        down = pos2
    else:
        up = pos2
        down = pos1
    if leght == 0:
        return
    angle = math.acos(dot((up - down), np.array([1, 0])) / leght) / math.pi * 180
    muscle = pygame.Surface((leght , width))
    muscle.set_colorkey((0, 0, 0))
    muscle.fill((100, 100, 100))
    rect = muscle.get_rect()
    m_cen = (pos1 + pos2) / 2
    rect.center = (WIDTH / 2 + m_cen[0] - cen[0], HEIGHT / 2 - m_cen[1] + cen[1])
    old_cen = rect.center
    new_image = pygame.transform.rotate(muscle, angle)
    rect = new_image.get_rect()
    rect.center = old_cen
    WIN.blit(new_image, rect)

def main():
    earth = Body()
    earth.nodes_pos = [[0.0, 0.0]]
    n = 16
    r = 200
    for i in range(n):
        x = r * np.cos(2 * i * np.pi / n)
        y = r * np.sin(2 * i * np.pi / n)
        earth.nodes_pos.append([x, y])
        earth.springs_nodes.append([0, i + 1])
        for ii in range(1):
            earth.springs_nodes.append([(i - ii - 1)%n + 1, i + 1])
            earth.springs_nodes.append([(i + ii + 1)%n + 1, i + 1])
    earth.springs_s = [0.1 for _ in earth.springs_nodes]
    print(earth.nodes_pos)
    earth.show_sim()
    plt.plot(earth.x, earth.data)
    plt.show()
    plt.plot(earth.x, earth.angle_data)
    plt.show()
    plt.plot(earth.data, earth.angle_data)
    plt.show()

if __name__ == "__main__":
    import pygame
    import matplotlib.pyplot as plt

    WIDTH, HEIGHT = 1400, 700
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("game")

    main()


