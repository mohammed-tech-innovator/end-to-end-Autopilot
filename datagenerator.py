from pynput import keyboard
import time
import os
import numpy as np
from mss import mss
import cv2
import torch


path = 'raw_data/train/'


screen_hight = 280
screen_width = 580
screen_top_padding = 150
screen_left_padding = 60
#################
Image_hight = 78
Image_width = 224


sct = mss()
monitor={'top':screen_top_padding,'left':screen_left_padding,'width':screen_width,'height':screen_hight}


forward_img_counter = 0
left_img_counter = 0
reverse_img_counter = 0
right_img_counter = 0
break_img_counter = 0


def load_status():
    global forward_img_counter
    global left_img_counter
    global reverse_img_counter
    global right_img_counter
    global break_img_counter

    try :
        state = torch.load(path + 'status.s')
        forward_img_counter = state['forword']
        left_img_counter = state['left']
        reverse_img_counter = state['reverse']
        right_img_counter = state['right']
        break_img_counter = state['break']

        print('state was loaded !!')
    except :
        print('state loading failed !!!')


    return

def save_status():

    state = {
    'forword' : forward_img_counter , 
    'left' : left_img_counter ,
    'reverse' : reverse_img_counter , 
    'right': right_img_counter , 
    'break' : break_img_counter ,

    }

    torch.save(state ,path + 'status.s')
    print("state was saved .")

    return


def show_state():
    print("forward_img_counter : ",forward_img_counter)
    print("left_img_counter : ",left_img_counter)
    print("reverse_img_counter : ",reverse_img_counter)
    print("right_img_counter : ",right_img_counter)
    print("break_img_counter : ",break_img_counter)
    print('total : ' + str(forward_img_counter + left_img_counter + reverse_img_counter + right_img_counter + break_img_counter))
    return







def grab_screen():
    screen = np.array(sct.grab(monitor))
    screen = cv2.resize(screen, (Image_width, Image_hight))
    return screen



def on_press(key):
    
    try:
        if (key.char == 'o'):
            global forward_img_counter
            forward_img_counter += 1
            screen = grab_screen()
            img_dir = path + 'forward/forward.' + str(forward_img_counter) + '.jpg'
            cv2.imwrite(img_dir, screen)
            
        if (key.char == 'a'):
            global left_img_counter
            left_img_counter += 1
            screen = grab_screen()
            img_dir = path + 'left/left.' + str(left_img_counter) + '.jpg'
            cv2.imwrite(img_dir, screen)
            
        if (key.char == 's'):
            global reverse_img_counter
            reverse_img_counter += 1
            screen = grab_screen()
            img_dir = path + 'reverse/reverse.' + str(reverse_img_counter) + '.jpg'
            cv2.imwrite(img_dir, screen)
            
        if (key.char == 'd'):
            global right_img_counter
            right_img_counter += 1
            screen = grab_screen()
            img_dir = path + 'right/right.' + str(right_img_counter) + '.jpg'
            cv2.imwrite(img_dir, screen)
        if (key.char == 'l'):
            global break_img_counter
            break_img_counter += 1
            screen = grab_screen()
            img_dir = path + 'break/break.' + str(break_img_counter) + '.jpg'
            cv2.imwrite(img_dir, screen)


    except AttributeError:
        if (key == keyboard.Key.esc):
            return False


if __name__ == "__main__":
    
    
    time.sleep(6)
    load_status()

    
    with keyboard.Listener(on_press) as listener:
        listener.join()



    show_state()
    save_status()


