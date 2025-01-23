import menu
import pygame as pg
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

def end(counter):
	# Initialize Pygame
	pg.init()

	# Set the dimensions of the window
	width, height = 1280, 720
	window = pg.display.set_mode((width, height))

	# Set the title of the window
	pg.display.set_caption("Here's your gifts!")

	# Define colors
	black = (0, 0, 0)  # Black color in RGB


	# Main loop
	running = True
	while running:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				running = False

		# Fill the window with red
		window.fill(black)

		# glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		# menu.drawText(580, 650, "Time's up!", (255, 0, 0))
		# menu.drawText(580, 650, "Gifts"+str(counter), (255, 0, 0))
		
		# Update the display
		pg.display.flip()

	# # Quit Pygame properly
	# pg.quit()
	# sys.exit()
