import time
import menu

def timer():
	# my_time = int(input("Enter the time in seconds: "))
	my_time = 180
	for x in range(my_time, 0, -1):
		seconds = x % 60
		minutes = int(x / 60) % 60
		menu.drawText(580, 650, f"{minutes:02}:{seconds:02}", (255, 0, 0))
		# print(f"{minutes:02}:{seconds:02}")
		time.sleep(1)

	print("TIME'S UP!")
