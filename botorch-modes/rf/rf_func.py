# bj rf test
from sklearn.ensemble import RandomForestClassifier
import timeit
from dataloader import load_data
#from dataloader import load_data

def rf_model(hyper_vector,machine_num,datasets_num):

	# pool of hypers
	trees = hyper_vector[0] +1
	features = ['auto', 'sqrt', 'log2']
	if (hyper_vector[2] == 0):
		depth = None
	else:
		depth = hyper_vector[2]
	split = hyper_vector[3]
	leaf = hyper_vector[4]
	fuc = ['gini', 'entropy']
	boots = ['True', 'False']

	# Build model 
	rf = RandomForestClassifier(n_estimators=trees,
								max_features=features[hyper_vector[1]], 
								max_depth=depth, 
								min_samples_split = split,
                    			min_samples_leaf =leaf, 
                    			criterion=fuc[hyper_vector[5]], 
                    			bootstrap=boots[hyper_vector[6]],
                                        random_state=0)

	# Training and Validating
	X_train = load_data(machine_num,datasets_num)[0]
	y_train = load_data(machine_num,datasets_num)[1]
	X_val = load_data(machine_num,datasets_num)[2]
	y_val = load_data(machine_num,datasets_num)[3]
	start = timeit.default_timer()
	rf.fit(X_train, y_train)
	score_val = rf.score(X_val, y_val)
	stop = timeit.default_timer()
	time = stop - start
	
	# Results
	#print('Training Time: ',stop - start)
	#print("Training set score: %f" % mlp.score(X_train, y_train))
	#print("Test set score: %f" % mlp.score(X_test, y_test))

	return time, score_val


