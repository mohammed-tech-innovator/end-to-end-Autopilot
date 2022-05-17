# End to End Learning for Self-Driving Cars
Directly maps the Images from the environment into a set of unique actions, the physical
state of the car acts as a memory to the system, external environmental factors will also
influence the physical state of the car, the whole system forms a closed-loop system, the
need for monitoring system is obvious to maintain the stability of the car, however, such
a monitoring system is closely related to the actual car used and out of the scope of this
work.
#  Model architecture
Two models influenced by NVidia End to End have been developed, the first one
utilized the VGG architecture while the second is based on the ResNests architecture
both of the two models uses Mish activation[36] and instance normalization
between convolutional layers, the two models maps the input images into four actions:
• Speedup
• Break
• Left
• Righ
# VGG-based model
composed of layers connected in series (single path from the input to the output), 6 convolutional layers followed by 5 fully-connected layers, 
Instance normalization is applied after each convolutional layer.
#  ResNets-based model
Consists of two convolutional layer followed by residual layers and finally fully connected
layers.
#  The dataset
 Grand Theft Auto Vice City (2002). New York, NY: Rockstar Games was
found to be a perfect candidate for the simulation environment; since the video game
wasn’t developed as a self-driving cars simulator we build a python program that captures
the images from the screen every time one of the keys associated with the four actions is
pressed, the program then rescale the image to 224x78 and save the image for the training
phase.
link : https://drive.google.com/file/d/1qsP3Hz2vY39LSvHtxNn9tQd9p4X9DOBy/view?usp=sharing
# actual performance 
ResNets-based Model : https://www.youtube.com/watch?v=Do-ODz6dHFw 

VGG-based Model : https://www.youtube.com/watch?v=Xf3BqoGuUOw
