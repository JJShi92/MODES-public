# bj mlp test
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import timeit
#from dataloader import load_data
from dataloader import load_data_eval as load_data
import numpy as np

def mlp_model_test(hyper_vector,machine_num,datasets_num):
	# pool of hypers
	layer = hyper_vector[0] + 1
	unit = hyper_vector[1]
	lay_unit = (unit,) * layer
	act = ['identity', 'logistic', 'tanh', 'relu']
	alh = hyper_vector[3]
	seed = 0
	ini = hyper_vector[4]

	# Build model 
	mlp = MLPClassifier(hidden_layer_sizes=lay_unit,
						max_iter=60, 
						alpha=alh, 
						activation = act[hyper_vector[2]],
                    	solver ='adam', 
                    	tol=1e-4, 
                    	learning_rate_init=ini,
                    	random_state=0,
                    	verbose=False)

	# Training and Validating
	X_train = load_data(machine_num,datasets_num)[0]
	y_train = load_data(machine_num,datasets_num)[1]
	X_val = load_data(machine_num,datasets_num)[4]
	y_val = load_data(machine_num,datasets_num)[5]
	start = timeit.default_timer()
	mlp.fit(X_train, y_train)
	score_val = mlp.score(X_val, y_val)
	stop = timeit.default_timer()
	time = stop - start
	
	# Results
	#print('Training Time: ',stop - start)
	#print("Training set score: %f" % mlp.score(X_train, y_train))
	#print("Test set score: %f" % mlp.score(X_test, y_test))

	return time, score_val
