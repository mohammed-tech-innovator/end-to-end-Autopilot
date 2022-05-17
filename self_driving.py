import autopilot_model as AM
import torch
import cv2
from mss import mss
from pynput import keyboard
import os
import time as t
import numpy as np
import matplotlib.pyplot as plt
import pyautogui


screen_hight = 280
screen_width = 580
screen_top_padding = 150
screen_left_padding = 60
#################
Image_hight = 78
Image_width = 224

sct = mss()
monitor={'top':screen_top_padding,'left':screen_left_padding,'width':screen_width,'height':screen_hight}



class AutoPilot():
	def __init__(self , path = '', delay = 0 , recorde_trip = False):

		self.pilot_model = AM.NeuralNet()
		try :

			device = torch.device('cpu')
			self.pilot_model = torch.load(os.path.join(path , 'results/AutoPilot.model'), map_location=device)
			print('model was founed and loaded succsessfully .')
		except:
			print('model not founed untrained model was loaded ')



		self.loop_delay = delay
		self.recorde_trip = recorde_trip
		self.trip = []

		return

	def grab_screen(self):
		screen = np.array(sct.grab(monitor))
		screen = cv2.resize(screen, (Image_width, Image_hight))
		return screen


	def drive(self,duration = 10):
		start = t.time()
		end = start

		while (end - start < duration) :
			timer = t.time()
			frame = self.grab_screen()
			frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

			if self.recorde_trip :

				self.trip.append(frame)



			frame = torch.tensor(frame , dtype = torch.float32).view(1,3,Image_width,Image_hight)

			raw_output = self.pilot_model(frame)
			output = torch.argmax(raw_output)
			if output == 0 :
				
				pyautogui.keyDown('o')
				t.sleep(self.loop_delay)
				pyautogui.keyUp('o')
				print('foward')

			elif output == 1 :
				pyautogui.keyDown('l')
				t.sleep(self.loop_delay)
				pyautogui.keyUp('l')
				print('backward')

			elif output == 2 :
				
				pyautogui.keyDown('a')
				#pyautogui.keyDown('o')
				t.sleep(self.loop_delay)
				#pyautogui.keyUp('o')
				pyautogui.keyUp('a')
				
				print('left')

			elif output == 3 :
				pyautogui.keyDown('d')
				#pyautogui.keyDown('o')
				t.sleep(self.loop_delay)
				#pyautogui.keyUp('o')
				pyautogui.keyUp('d')
				
				print('right')



			end = t.time()

			
			print(end - timer)


		return


	def showTrip(self):
		for frame in self.trip :
			plt.clf()
			plt.imshow(frame)
			plt.draw()
			plt.pause(self.loop_delay)

		print('done')

		return








a = AutoPilot(delay = 0.0 , recorde_trip = True)
t.sleep(10)
a.drive(duration = 180)
t.sleep(5)
#a.showTrip()oddd