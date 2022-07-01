
import torch
import torch.nn as nn
from   torch.autograd import Variable
import numpy    as np
import pandas
import models as my_model
from scipy import stats

### Here I am doing initialization
CSVTraining = pandas.read_csv('Dataset/train.csv')
CSVTest = pandas.read_csv('Dataset/test.csv')
print("Initialization step")
N_training = CSVTraining.values.shape[0]
N_testing = CSVTest.values.shape[0]
numb_columns = CSVTraining.values.shape[1] - 12
initial_train = np.zeros((N_training, 3))
initial_test  = np.zeros((N_testing, 3))
x_training = np.zeros((N_training, numb_columns))
y_training = np.zeros((N_training, 4))
x_test = np.zeros((N_testing,  numb_columns))

modeltype = "neural_1"
inpdim     = numb_columns
outdim    = 4
learnrate = 0.01
epochs       = 8500

for j in range(0, 4):
	y_training[:,j] =  CSVTraining.values[:,j + 9]

for j in range(0 , numb_columns - 5):
	x_training[:,j] = stats.zscore(CSVTraining.values[:,j + 17])

x_training[:,numb_columns - 4] = stats.zscore(CSVTraining.values[:,6])
x_training[:,numb_columns - 5] = stats.zscore(CSVTraining.values[:,7])

print ("Reading step")

for j in range(0 , numb_columns - 5):
	x_test[:,j]     = stats.zscore(CSVTest.values[:,j + 13])

x_test[:,numb_columns - 4] = stats.zscore(CSVTest.values[:,6])
x_test[:,numb_columns - 5] = stats.zscore(CSVTest.values[:,7])



print ("Reading step ")

for i in range(N_training):
	if i != 1185:
		for j in range(0, 3):
			A = CSVTraining.values[i][14 + j].strip('[')
			initial_train[i][j] = np.sqrt(np.dot(np.fromstring(A.strip(']'), dtype = float, sep = ','), np.fromstring(A.strip(']'), dtype = float, sep = ',')))
		print(i)

for i in range(N_testing):
	for j in range(0, 3):
		A = CSVTest.values[i][10 + j].strip('[')
		initial_test[i][j] = np.sqrt(np.dot(np.fromstring(A.strip(']'), dtype = float, sep = ','), np.fromstring(A.strip(']'), dtype = float, sep = ',')))
	print(i)
#### standardizing parameters
for j in range(numb_columns - 3, numb_columns):
	x_training[:,j] = stats.zscore(initial_train[:,j - numb_columns + 3])

for j in range(numb_columns - 3, numb_columns):
	x_test[:,j] = stats.zscore(initial_test[:,j - numb_columns + 3])

print ("Training step ")

##Neural network, gradient, and loss function choice
if modeltype == "linearRegression":
	model    = my_model.linearRegression(inpdim, outdim)
elif modeltype == "neural":
	model    = my_model.neural(inpdim, outdim)
elif modeltype == "neural_1":
	model    = my_model.neural_1(inpdim, outdim)


#lossfn = nn.MSELoss()
#lossfn = nn.KLDivLoss()
lossfn = nn.SmoothL1Loss()
#lossfn = nn.SoftMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learnrate, amsgrad=True)

inputdata = Variable(torch.from_numpy(x_training).float())
labelvalues = Variable(torch.from_numpy(y_training).float())

#Model Training step
for epoch in range(0,epochs):
	optimizer.zero_grad()
	outputs = model(inputdata)
	loss = lossfn(outputs, labelvalues)
	loss.backward()
	# update parameters
	optimizer.step()
	if epoch % 100 == 0:
		print('epoch number {}, loss {}'.format(epoch, loss.item()))

# Testing step
with torch.no_grad():
	pred_vals = model(Variable(torch.from_numpy(x_test)).float()).data.numpy()

output_file = open('prediction.csv', 'w')
output_file.write('id,Predicted\n')

for i in range(0, N_testing):
	output_file.write('test_' + str(i) + '_val_error,')
	output_file.write('{}\n'.format(pred_vals[i][0]))
	output_file.write('test_' + str(i) + '_train_error,')
	output_file.write('{}\n'.format(pred_vals[i][2]))
