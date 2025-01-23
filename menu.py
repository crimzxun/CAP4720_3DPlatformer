import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def draw_menu(screen):

    mouse = pg.mouse.get_pos()
    click = pg.mouse.get_pressed()

    white = (255, 255, 255)
    green = (0, 200, 0)
    red = (255, 0, 0)
    black = (0, 0, 0)
    grey = (100, 100, 100)

    screen.fill(green)

    # MENU BACKGROUND
    rect_width = 600
    rect_height = 400

    # Center the menu screen
    screen_width, screen_height = screen.get_size()
    menu_x = (screen_width - rect_width) // 2
    menu_y = (screen_height - rect_height) // 2 

    rect = pg.Rect(menu_x, menu_y, rect_width, rect_height)
    pg.draw.rect(screen, white, rect)

    font = pg.font.Font(None, 50)
    text_surface = font.render("Merry Christmas", True, red)
    text_rect = text_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 - 120))
    screen.blit(text_surface, text_rect)


    # BUTTONS
    button_width = 80
    button_height = 30

    button_x = (screen_width - button_width) // 2
    button_y = (screen_height - button_height) // 2
    start_button = pg.Rect(button_x, button_y, button_width, button_height)
    pg.draw.rect(screen, black, start_button)

    quit_button = pg.Rect(button_x, button_y + 70, button_width, button_height)
    pg.draw.rect(screen, black, quit_button)

    # MOUSE ON BUTTON
    if start_button.collidepoint(mouse):  # Check if mouse is over the Start button
        pg.draw.rect(screen, grey, start_button)
        if click[0] == 1:
            screen.fill((0, 0, 0))
            return True

    if quit_button.collidepoint(mouse):  # Check if mouse is over the Quit button
        pg.draw.rect(screen, grey, quit_button)
        if click[0] == 1:  # Check if left mouse button is clicked
            pg.quit()  # Quit Pygame


    font = pg.font.Font(None, 25)

    # BUTTONS' TEXT
    start_text = font.render("Start", True, white)
    start_text_rect = start_text.get_rect(center=start_button.center)
    screen.blit(start_text, start_text_rect)

    quit_text = font.render("Quit", True, white)
    quit_text_rect = quit_text.get_rect(center=quit_button.center)
    screen.blit(quit_text, quit_text_rect)



    return False

    

def drawText(x, y, text, color):   
        
    font = pg.font.Font(None, 50)                                 
    textSurface = font.render(text, True, color).convert_alpha()
    textData = pg.image.tostring(textSurface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


def draw_result(screen, counter):
    mouse = pg.mouse.get_pos()
    click = pg.mouse.get_pressed()

    white = (255, 255, 255)
    green = (0, 200, 0)
    red = (255, 0, 0)
    black = (0, 0, 0)
    grey = (100, 100, 100)

    # Clear the screen with a green background
    screen.fill(green)

    # MENU BACKGROUND
    rect_width = 600
    rect_height = 400

    # Center the menu screen
    screen_width, screen_height = screen.get_size()
    menu_x = (screen_width - rect_width) // 2
    menu_y = (screen_height - rect_height) // 2

    rect = pg.Rect(menu_x, menu_y, rect_width, rect_height)
    pg.draw.rect(screen, white, rect)

    font = pg.font.Font(None, 50)
    text_surface = font.render("Score: " + counter, True, red)
    text_rect = text_surface.get_rect(center=(screen_width // 2, screen_height // 2 - 120))


    f = open("highscore.txt", "r")
    pb = " ".join(line.rstrip() for line in f)

    print("this is your pb", pb)
    f.close()
    if(int(pb) < int(counter)):
        pb = counter
        f = open("highscore.txt", "w")
        f.write(counter)
        f.close()


    text_high = font.render("Personal Best: " + pb, True, red)
    text_rect2 = text_high.get_rect(center=(screen_width // 2, screen_height // 2 - 50))

    screen.blit(text_surface, text_rect)
    screen.blit(text_high,text_rect2)

    # BUTTONS
    button_width = 80
    button_height = 30

    button_x = (screen_width - button_width) // 2
    button_y = (screen_height - button_height) // 2


    quit_button = pg.Rect(button_x, button_y + 70, button_width, button_height)
    pg.draw.rect(screen, black, quit_button)



    if quit_button.collidepoint(mouse):  # Check if mouse is over the Quit button
        pg.draw.rect(screen, grey, quit_button)
        if click[0] == 1:  # Check if left mouse button is clicked
            pg.quit()  # Quit Pygame

    font = pg.font.Font(None, 25)


    quit_text = font.render("Quit", True, white)
    quit_text_rect = quit_text.get_rect(center=quit_button.center)
    screen.blit(quit_text, quit_text_rect)

    return False