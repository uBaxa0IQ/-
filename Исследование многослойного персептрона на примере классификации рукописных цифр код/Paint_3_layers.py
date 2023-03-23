import pygame
import tensor as tf
from file3_6_weights0_1 import weights0_1
from file3_6_weights1_2 import weights1_2

pygame.init()

Hight = 1100
Lenght = 840
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode([Hight, Lenght])
screen.fill(WHITE)
pygame.draw.rect(screen, RED, (840, 0, 2, 840))
pygame.display.update()

font_name = pygame.font.match_font('arial')

def draw_text(surf, text, size, x, y, color):
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)

running = True

drowing = False


mass = tf.create_zero_matrix(28, 28)


while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            drowing = not(drowing)
        if drowing  :
            if event.type == pygame.MOUSEMOTION and event.pos[0] < 840:
                mass[event.pos[1] // 30][event.pos[0] // 30] = 1
                pygame.draw.rect(
                    screen, BLACK,
                    (event.pos[0] // 30 * 30,
                        event.pos[1] // 30 * 30,
                        30, 30))
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            screen.fill(WHITE)
            pygame.draw.rect(screen, RED, (840, 0, 2, 840))
            mass = tf.create_zero_matrix(28, 28)

        pygame.display.update()




    layer_0 = tf.matrix2_to_1(mass)
    layer_1 = tf.relu(tf.scal_mul(layer_0, weights0_1))
    layer_2 = tf.softmax(tf.scal_mul(layer_1, weights1_2))
    pygame.draw.rect(screen, WHITE, (842, 0, 258, 840))
    for i in range(len(layer_2)):
        if i == tf.mass_argmax(layer_2):
            color = RED
        else:
            color = BLACK
        draw_text(screen, str(i) + ' : ' + str(round(layer_2[i] * 100, 1)), 50, 970, Lenght / 10 * i, color)




pygame.quit()