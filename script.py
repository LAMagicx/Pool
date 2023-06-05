from itertools import combinations as C
import pygame, sys
import numpy as np
import vector, line


def collide(pos, r, vel, C):
    if vel == 0:
        raise Exception("vel is 0", vel)
    t = (C - (pos+r)) / vel
    if t > 1: 
        raise Exception("t is > 1", t)
    # collision_point
    pos = pos + t * vel
    # change vel
    vel *= -(1-t)
    return pos, vel


def wall_collision(pos, r, vel, dt, w, h):
    pos_n = pos + vel * dt
    # detect each wall collision
    if pos_n[0]-r < 0:
        # hit left
        pos[0], vel[0] = collide(pos[0], -r, vel[0], 0)
    if pos_n[0]+r > w:
        # hit right
        pos[0], vel[0] = collide(pos[0], r, vel[0], w)
    if pos_n[1]-r < 0:
        # hit up
        pos[1], vel[1] = collide(pos[1], -r, vel[1], 0)
    if pos_n[1]+r > h:
        # hit down
        pos[1], vel[1] = collide(pos[1], r, vel[1], h)

    return pos, vel


def collide_balls(N, poss, r, vels, dt):
    """detects collision between balls"""
    res = vector.init(0, 0).astype(np.float64)
    collided = False
    for i,j in list(C(range(N), 2)):
        p1 = poss[i]
        v1 = vels[i]
        p2 = poss[j]
        v2 = vels[j]
        d = vector.dist_quick(p1, p2)
        if d < (2*r)**2:
            # collision

            # masses
            m1, m2 = 1, 1
            n = p2 - p1
            un = n / vector.norm(n)
            ut = vector.init(-un[1], un[0])
            v1n = vector.dot(un, v1)
            v1t = vector.dot(ut, v1)
            v2n = vector.dot(un, v2)
            v2t = vector.dot(ut, v2)
            v1n_ = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2)
            v2n_ = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2)
            v1n_ = un * v1n_
            v1t_ = ut * v1t
            v2n_ = un * v2n_
            v2t_ = ut * v2t
            vel1 = v1n_ + v1t_
            vel2 = v2n_ + v2t_
            vels[i] = vel1
            vels[j] = vel2
            poss[i] = p1 - un * (r-np.sqrt(d)/2)
            poss[j] = p2 + un * (r-np.sqrt(d)/2)


# pygame setup
screen_width, screen_height = 750, 750
FPS = 30
pygame.display.init()
screen = pygame.display.set_mode((screen_width, screen_height), pygame.DOUBLEBUF)
clock = pygame.time.Clock()

r = 20
N = 15
T = 10
dt = 1/T
scl = 5
mouse_x, mouse_y = -1, -1
ball_positions = np.array([vector.random(r*2, min(screen_width, screen_height)-r*2) for i in range(N)],dtype=np.float32)
ball_velocities = np.array([vector.init(0,0) for i in range(N)],dtype=np.float32)
ball_accelerations = np.array([vector.random()*scl for i in range(N)],dtype=np.float32)

line_positions = np.array([line.init(0, 0, 0, screen_height),
                           line.init(0, screen_height, screen_width, screen_height),
                           line.init(screen_width, screen_height, screen_width, 0),
                           line.init(screen_width, 0, 0, 0)]
                           )

mouse_down = False
mouse_click_x, mouse_click_y = -1, -1
mouse_release_x, mouse_release_y = -1, -1
force_x, force_y = -1, -1
force = 2
force_limit = 5
selected = -1

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
            if selected != -1:
                center_pos = ball_positions[selected]
                force_vec = (center_pos - np.array([mouse_release_x, mouse_release_y]))/force
                ball_accelerations[selected] = vector.limit(force_vec, force_limit)
                selected = -1

    screen.fill((51,51,51))
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

        ## do ball collisions
        collide_balls(N, ball_positions, r, ball_velocities, dt)

        ## Upate ball positions
        ball_velocities += ball_accelerations
        ball_positions += ball_velocities * dt
        ball_accelerations = np.zeros(shape=(N,2))
        ball_velocities *= 1 - (0.02 * dt)
    
    ## draw predict_line
    if selected != -1:
        x,y = 2 * ball_positions[selected] - np.array([mouse_x, mouse_y])
        pygame.draw.line(screen, (10, 10, 10), ball_positions[selected].astype(int), [x, y], 1)

    v = vector.add()

    ## draw balls
    index = 0
    for px, py in ball_positions:
        pygame.draw.circle(screen, (200, 200, 200), [int(px), int(py)], r)
        if vector.dist([px, py], [mouse_x, mouse_y]) < r:
            if mouse_down:
                if selected == -1:
                    selected = index
            else:
                pygame.draw.circle(screen, (200, 100, 100), [int(px), int(py)], r, 3)
        if selected == index:
            ball_velocities[index] = np.zeros(shape=(2,))
            pygame.draw.circle(screen, (220, 50, 50), [int(px), int(py)], r, 3)
        index += 1


    pygame.display.flip()
    clock.tick(FPS)


print("quitting")
pygame.display.quit()
exit()
sys.exit()
