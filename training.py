import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch.optim as optim
import os


import autopilot_model
import torch

device =  torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')


training_set_size = 0.8
test_set_size = 1.0 - training_set_size # percentage
LR = 0.002
Epochs = 10
Rounds = 10
load_from_check_point = False
path = 'data/'
results_path = 'drive/MyDrive/'
data = []
training_loss = []
validation_loss = []
net = autopilot_model.NeuralNet()


optimizer = optim.Adam(net.parameters(),lr = LR)


if load_from_check_point :
	#optimizer = torch.load(os.path.join(results_path , 'results/optimizer.op'))
	#status = torch.load(os.path.join(results_path  , 'results/status.s'))
	#training_loss = status['training_loss']
	#validation_loss = status['validation_loss']
	net = torch.load(results_path + 'results/AutoPilot.model',map_location=torch.device('cpu'))
	net.eval()


net = net.to(device)


data_files = ['DATA_2.dt' , 'DATA_1.dt' ,'DATA_3.dt' , 'DATA_4.dt']






for round in range(Rounds):
	print("round number :" , round)

	loss_record = 10e9

	for data_file in data_files :
		print("using data file : " + data_file)
		del data
		content = torch.load(os.path.join(path , data_file))
		data = content['data']
		Forward = content['forward']
		Right = content['right']
		Left = content['left']
		Break = content['break']
		
		del content


		maximum = max(Forward , Right , Left , Break)

		w = torch.tensor([maximum/Forward , maximum/Right , maximum/Left , maximum/Break])

		w = (w/torch.sum(w)).to(device)

		print("data loaded cross entropy wights are : " , w)
		num_of_training_patches = int(training_set_size*len(data))
		num_of_test_patches = len(data) - num_of_training_patches
		randkey = list(range(len(data)))


		for epoch in range(Epochs) :
			random.shuffle(randkey) # random selection of the training and validation patches
			num_of_courr = 0
			training_index = randkey[:num_of_training_patches]
			test_index = randkey[num_of_training_patches:]
			tain_total_loss = 0
			test_total_loss = 0
			print("############################################")
			print("Epoch : ",epoch," start training over ",num_of_training_patches,"patches.")
			for index in tqdm(training_index) :

				try :
					net , loss = autopilot_model.train(net , data[index] ,optimizer , weight = w)
					tain_total_loss = tain_total_loss + loss.cpu().detach().numpy()
				except :
					num_of_courr = num_of_courr + 1


			print()
			print("training phase is over with total loss : ",tain_total_loss ,"Avg loss : ",(tain_total_loss/num_of_training_patches))
			print("start testing over ",num_of_test_patches,"patches.")
			print()

			for index in tqdm(test_index) :
				try:
					loss = autopilot_model.test(net , data[index] ,  weight = w)
					test_total_loss = test_total_loss + loss.cpu().detach().numpy()
				except :
					num_of_courr = num_of_courr + 1

			print()
			print("testing phase is over with total loss : ",test_total_loss ,"Avg loss : ",(test_total_loss/num_of_test_patches))
			training_loss.append((tain_total_loss/num_of_training_patches))
			validation_loss.append((test_total_loss/num_of_test_patches))
			print('num of courrupted batches = ',num_of_courr)

			#save check a point
			if test_total_loss < loss_record :
				torch.save(net,results_path + 'results/AutoPilot.model')
				torch.save(optimizer ,results_path + 'results/optimizer.op')
				status = 
				{
				'training_loss' : training_loss , 
				'validation_loss' : validation_loss,
				}
				torch.save(status , results_path + 'results/status.s')
				print('check point saved !')
				loss_record = test_total_loss
		


plt.plot(list(range(len(training_loss))) , training_loss , 'r')
plt.plot(list(range(len(validation_loss))) , validation_loss , 'b')
plt.show()