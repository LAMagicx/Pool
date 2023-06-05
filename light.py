import pygame, sys
import numpy as np
import vector, line

# pygame setup
screen_width, screen_height = 750, 750
FPS = 0
pygame.display.init()
screen = pygame.display.set_mode((screen_width, screen_height), pygame.DOUBLEBUF)
clock = pygame.time.Clock()

B = 100
r = 1
N = 1
T = 2
dt = 1/T
scl = 1
mouse_x, mouse_y = -1, -1
#ball_positions = np.array([vector.random(r*2, min(screen_width, screen_height)-r*2) for i in range(N)],dtype=np.float32)
ball_positions = np.array([vector.init(250,400) for i in range(N)],dtype=np.float32)
ball_velocities = np.array([vector.init(0,0) for i in range(N)],dtype=np.float32)
ball_accelerations = np.array([vector.random()*scl for i in range(N)],dtype=np.float32)
#ball_accelerations = np.array([vector.init(0.001, -1)*scl for i in range(N)],dtype=np.float32)


line_positions = np.array([line.init(0, 0, 0, screen_height),
                           line.init(0, screen_height, screen_width, screen_height),
                           line.init(screen_width, screen_height, screen_width, 0),
                           line.init(screen_width, 0, 0, 0),
                           line.init(0, 0, 100, 700),
                           line.init(200,200,300,300),
                           line.init(500,500,300,600)])

loop = True
while loop:
    for event in pygame.event.get():
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                loop = False
        if event.type == pygame.QUIT:
            loop = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
            mouse_click_x, mouse_click_y = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
            mouse_release_x, mouse_release_y = pygame.mouse.get_pos()

    #screen.fill((51,51,51))
    lines = []
    for _ in range(B):
        for _ in range(T):
            ## do wall collision
            for i in range(N):
                for l in line_positions:
                    l2 = np.array([ball_positions[i], ball_positions[i] + dt * ball_velocities[i]])
                    intersect, s, t = line.intersect(l, l2)
                    if 0 <= s <= 1 and 0 <= t <= 1:
                        s1 = line.vector(l)
                        s1n = s1 / vector.norm(s1)
                        ut = vector.init(-s1n[1], s1n[0])
                        s2 = ball_velocities[i]
                        s3 = s2 - 2 * vector.dot(s2, ut) * ut
                        ball_positions[i] = intersect + (1-t) * s3
                        ball_velocities[i] = s3
                lines.append((int(ball_positions[i][0]), int(ball_positions[i][1])))



            ## Upate ball positions
            ball_velocities += ball_accelerations
            ball_positions += ball_velocities * dt
            ball_accelerations = np.zeros(shape=(N,2))

    pygame.draw.lines(screen, (200, 200, 200), False, lines, 1)
    ## draw lines
    for l in line_positions:
        pygame.draw.line(screen, (100, 100, 100), l[0], l[1], 2)
    
    ## draw balls
    for px, py in ball_positions:
        pygame.draw.circle(screen, (200, 200, 200), [int(px), int(py)], r)



    pygame.display.flip()
    clock.tick(FPS)

pygame.display.quit()

