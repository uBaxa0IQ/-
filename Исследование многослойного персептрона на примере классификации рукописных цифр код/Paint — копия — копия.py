import pygame
import tensor as tf
from file4_5_weights0_1 import weights0_1
from file4_5_weights1_2 import weights1_2
from file4_5_weights2_3 import weights2_3

pygame.init()

Hight = 25 * 28
Lenght = 25 * 28
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode([Hight, Lenght])
screen.fill(WHITE)
pygame.draw.rect(screen, RED, (Lenght, 0, 2, Lenght))
pygame.display.update()



running = True





k = 0

while running:


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:

            #layer_3 = [0] * 10
            #layer_3[k] = 1

            layer_2 = [0] * 10
            layer_2[k] = 1

            #layer_2 = tf.scal_mul(layer_3, tf.reverse(weights2_3))
            layer_1 = tf.scal_mul(layer_2, tf.reverse(weights1_2))
            layer_0 = tf.scal_mul(layer_1, tf.reverse(weights0_1))

            new_layer_0 = [([0] * 28).copy() for i in range(28)]
            for i in range(28):
                for j in range(28):
                    new_layer_0[i][j] = layer_0[28 * i + j]

            layer_0 = new_layer_0

            for i in range(len(layer_0)):
                for j in range(len(layer_0[i])):
                    if layer_0[i][j] >= 0:
                        COLOR = (255 - int(layer_0[i][j] * 255), 255 - int(layer_0[i][j] * 255), 255 - int(layer_0[i][j] * 255))
                    else:
                        COLOR = (255, 255, 255)
                    pygame.draw.rect(
                        screen, COLOR,
                        (j * 25,
                         i * 25,
                         25, 25))

            k += 1


        pygame.display.update()






pygame.quit()