from sklearn import datasets
from sklearn import svm    			
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
from sklearn import metrics
from scipy.fftpack import fft

X_train = np.load("x2.npy")
Y_train = np.load("y2.npy")

X_raw = X_train
Y_raw = Y_train
X_test = np.empty((200,2,1000))
Y_test = np.empty((200,1))

for i in range(int(0.05*len(Y_train))):

	rand_int1 = random.randint(0,int(0.25*len(X_train))-i)
	X_test[i,0,:] = X_train[rand_int1][0]
	X_test[i,1,:] = X_train[rand_int1][1]
	Y_test[i,:] = Y_train[rand_int1]
	X_train = np.delete(X_train, rand_int1, 0)
	Y_train = np.delete(Y_train, rand_int1, 0)

	rand_int2 = random.randint(int(0.25*len(X_train))-i, int(0.5*len(X_train))-i)
	X_test[i+50,0,:] = X_train[rand_int2][0]
	X_test[i+50,1,:] = X_train[rand_int2][1]
	Y_test[i+50,:] = Y_train[rand_int2]
	X_train = np.delete(X_train, rand_int2, 0)
	Y_train = np.delete(Y_train, rand_int2, 0)

	rand_int3 = random.randint(int(0.5*len(X_train))-i, int(0.75*len(X_train))-i)
	X_test[i+100,0,:] = X_train[rand_int3][0]
	X_test[i+100,1,:] = X_train[rand_int3][1]
	Y_test[i+100,:] = Y_train[rand_int3]
	X_train = np.delete(X_train, rand_int3, 0)
	Y_train = np.delete(Y_train, rand_int3, 0)

	rand_int4 = random.randint(int(0.75*len(X_train))-i, len(X_train)-i)
	X_test[i+150,0,:] = X_train[rand_int3][0]
	X_test[i+150,1,:] = X_train[rand_int3][1]
	Y_test[i+150,:] = Y_train[rand_int4]
	X_train = np.delete(X_train, rand_int4, 0)
	Y_train = np.delete(Y_train, rand_int4, 0)

'''

Visualizing each tY_trainpe of voltage signal

'''

# sample1 = X_train[2]
# time1 = sample1[1]
# volt1 = sample1[0]

# sample2 = X_test[150]
# time2 = sample2[1]
# volt2 = sample2[0]

# sample3 = X_train[500]
# time3 = sample3[1]
# volt3 = sample3[0]

# sample4 = X_train[600]
# time4 = sample4[1]
# volt4 = sample4[0]

# plt.plot(time1, volt1)
# plt.show()

# plt.scatter(time2, volt2)
# plt.show()

# # plt.plot(time3, volt3)
# # plt.show()

# # plt.plot(time4, volt4)
# # plt.show()

X_train_feat_density = np.array([])
X_test_feat_density = np.array([])

'''

This is where the manipulation of the waves takes place. We have to do two of each for training and testing, not because
it is efficient but to show my TA the training/testing split.

'''

#find average of waves and mean


X_train_feat_stdv = np.array([])
X_test_stdv = np.array([])

for i in range(len(Y_train)):
	sample_i = X_train[i]
	volt_avg_i = np.mean(sample_i[0])
	volt_stdv = np.std(sample_i[0])
	X_train_feat_density = np.append(X_train_feat_density, [volt_avg_i], axis = 0)
	X_train_feat_stdv = np.append(X_train_feat_stdv, [volt_stdv], axis = 0)

for i in range(len(Y_test)):
	sample_i = X_test[i]
	volt_avg_i = np.mean(sample_i[0])
	volt_stdv = np.std(sample_i[0])
	X_test_feat_density = np.append(X_test_feat_density, [volt_avg_i], axis = 0)
	X_test_feat_stdv = np.append(X_train_feat_stdv, [volt_stdv], axis = 0)

#find peaks/crests and then find frequency

X_train_avg_frequency = np.array([])

X_train_fft = np.array([])
for i in range(len(Y_train)):
	sample_i = X_train[i]
	time_i = np.array([])
	for j in range(999):
		if ((sample_i[0][j] > sample_i[0][j-1]) and (sample_i[0][j] > sample_i[0][j+1])):
			time_i = np.append(time_i, [sample_i[1][j]], axis = 0)
	
	#Fourier transforms
	fft = np.mean(np.fft.fft(sample_i[0]))
	n = sample_i[0].size
	#freq = np.mean(np.fft.fftfreq(n, d = 0.0990990990990992))

	X_train_fft = np.append(X_train_fft,[fft.real])
	
	length = len(time_i)
	frequency = (1000/length)**2
	X_train_avg_frequency = np.append(X_train_avg_frequency, [frequency])

# plt.plot(X_train_fft)
# plt.show()
plt.scatter(Y_train, X_train_fft)
plt.show()

X_test_avg_frequency = np.array([])
X_test_fft = np.array([])

for i in range(len(Y_test)):
	sample_i = X_test[i]
	time_i = np.array([])
	for j in range(999):
		if ((sample_i[0][j] > sample_i[0][j-1]) and (sample_i[0][j] > sample_i[0][j+1])):
			time_i = np.append(time_i, [sample_i[1][j]], axis = 0)
			
	length = len(time_i)
	frequency = (1000/length)**2
	X_test_avg_frequency = np.append(X_test_avg_frequency, [frequency])
	#fourier trainforms
	fft = np.mean(np.fft.fft(sample_i[0]))
	X_test_fft = np.append(X_test_fft, [fft.real])

#amount changed between each point#

X_train_feat_pts_diff = np.array([])

for i in range(len(Y_train)):
	sample_i = X_train[i]
	pts_difference = np.array([])
	for j in range(999):
		pts_difference = np.append(pts_difference, [(abs(sample_i[0][j]-sample_i[0][j+1]))**3], axis = 0)
		
	sample_average = np.mean(pts_difference)
	X_train_feat_pts_diff = np.append(X_train_feat_pts_diff, [sample_average])

X_test_feat_pts_diff = np.array([])
for i in range(len(Y_test)):
	sample_i = X_test[i]
	pts_difference = np.array([])
	for j in range(999):
		pts_difference = np.append(pts_difference, [(abs(sample_i[0][j]-sample_i[0][j+1]))**4], axis = 0)
		
	sample_average = np.mean(pts_difference)
	X_test_feat_pts_diff = np.append(X_test_feat_pts_diff, [sample_average])

# print("average", np.shape(X_train_feat_density))
# print("Pts diff", np.shape(X_train_feat_pts_diff))

#Standard Deviation

# plt.scatter(X_train_fft, X_train_feat_stdv)
# plt.show()

# plt.scatter(X_train_fft, X_train_feat_density)
# plt.show()

# plt.scatter(X_train_fft, X_train_avg_frequency)
# plt.show()

'''

Here I did some brutal building of matrices because I couldnt figure out in-house functions to do that

'''
Y_test = np.squeeze(Y_test)
X_train_feat_stdv = np.transpose(X_train_feat_stdv)
X_train_feat_density = np.transpose(X_train_feat_density)
X_train_avg_frequency = np.transpose(X_train_avg_frequency)
X_train_fft = np.transpose(X_train_fft)


#Re-formatting arrays to match model requirements
X_train_feat_stdv = np.expand_dims(X_train_feat_stdv, axis = 0)
X_train_feat_stdv = np.transpose(X_train_feat_stdv)
X_train_feat_density = np.expand_dims(X_train_feat_density, axis = 0)
X_train_feat_density = np.transpose(X_train_feat_density)
X_train_avg_frequency = np.expand_dims(X_train_avg_frequency, axis = 0)
X_train_avg_frequency = np.transpose(X_train_avg_frequency)
X_train_feat_pts_diff = np.expand_dims(X_train_feat_pts_diff, axis = 0)
X_train_feat_pts_diff = np.transpose(X_train_feat_pts_diff)
X_train_fft = np.expand_dims(X_train_fft, axis = 0)
X_train_fft = np.transpose(X_train_fft)

X_train_feat_density_and_X_train_fft = np.array([X_train_feat_density, X_train_fft])
X_train_feat_density_and_X_train_fft = np.squeeze(X_train_feat_density_and_X_train_fft)
X_train_feat_density_and_X_train_fft = np.transpose(X_train_feat_density_and_X_train_fft)

X_train_feat_density_and_X_train_avg_frequency = np.array([X_train_feat_density, X_train_avg_frequency])
X_train_feat_density_and_X_train_avg_frequency = np.squeeze(X_train_feat_density_and_X_train_avg_frequency)
X_train_feat_density_and_X_train_avg_frequency = np.transpose(X_train_feat_density_and_X_train_avg_frequency)

X_train_feat_density_and_pts_diff_avg = np.array([X_train_feat_density, X_train_feat_pts_diff])
X_train_feat_density_and_pts_diff_avg = np.squeeze(X_train_feat_density_and_pts_diff_avg)
X_train_feat_density_and_pts_diff_avg = np.transpose(X_train_feat_density_and_pts_diff_avg)

X_train_feat_stdv_and_pts_diff_avg = np.array([X_train_feat_stdv, X_train_feat_pts_diff])
X_train_feat_stdv_and_pts_diff_avg = np.squeeze(X_train_feat_stdv_and_pts_diff_avg)
X_train_feat_stdv_and_pts_diff_avg = np.transpose(X_train_feat_stdv_and_pts_diff_avg)

#Testing

X_test_feat_stdv = np.transpose(X_test_feat_stdv)
X_test_feat_density = np.transpose(X_test_feat_density)
X_test_avg_frequency = np.transpose(X_test_avg_frequency)
X_test_fft = np.transpose(X_test_fft)

#Re-formatting arrays to match model requirements
X_test_feat_stdv = np.expand_dims(X_test_feat_stdv, axis = 0)
X_test_feat_stdv = np.transpose(X_test_feat_stdv)
X_test_feat_density = np.expand_dims(X_test_feat_density, axis = 0)
X_test_feat_density = np.transpose(X_test_feat_density)
X_test_avg_frequency = np.expand_dims(X_test_avg_frequency, axis = 0)
X_test_avg_frequency = np.transpose(X_test_avg_frequency)
X_test_feat_pts_diff = np.expand_dims(X_test_feat_pts_diff, axis = 0)
X_test_feat_pts_diff = np.transpose(X_test_feat_pts_diff)
X_test_fft = np.expand_dims(X_test_fft, axis = 0)
X_test_fft = np.transpose(X_test_fft)

X_test_feat_density_and_X_test_fft = np.array([X_test_feat_density, X_test_fft])
X_test_feat_density_and_X_test_fft = np.squeeze(X_test_feat_density_and_X_test_fft)
X_test_feat_density_and_X_test_fft = np.transpose(X_test_feat_density_and_X_test_fft)

X_test_feat_density_and_X_test_avg_frequency = np.array([X_test_feat_density, X_test_avg_frequency])
X_test_feat_density_and_X_test_avg_frequency = np.squeeze(X_test_feat_density_and_X_test_avg_frequency)
X_test_feat_density_and_X_test_avg_frequency = np.transpose(X_test_feat_density_and_X_test_avg_frequency)

X_test_feat_density_and_pts_diff_avg = np.array([X_test_feat_density, X_test_feat_pts_diff])
X_test_feat_density_and_pts_diff_avg = np.squeeze(X_test_feat_density_and_pts_diff_avg)
X_test_feat_density_and_pts_diff_avg = np.transpose(X_test_feat_density_and_pts_diff_avg)

X_test_feat_stdv_and_pts_diff_avg = np.array([X_test_feat_stdv, X_test_feat_pts_diff])
X_test_feat_stdv_and_pts_diff_avg = np.squeeze(X_test_feat_stdv_and_pts_diff_avg)
X_test_feat_stdv_and_pts_diff_avg = np.transpose(X_test_feat_stdv_and_pts_diff_avg)


print("test", X_train_feat_density_and_X_train_avg_frequency[0])
print(X_train_feat_density_and_X_train_fft[0])

'''

This is where I actuallY_train performed the SVM. I used the stock code from the website we looked at.

'''

'''
SVM#1 X_train_feat_density and X_train_avg_frequency
'''

C = 1.0
svc = svm.SVC(kernel = 'linear', C=1.0).fit(X_train_feat_density_and_X_train_fft, Y_train)
lin_svc = svm.LinearSVC(C=1.0).fit(X_train_feat_density_and_X_train_fft, Y_train)
rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.7, C=1.0).fit(X_train_feat_density_and_X_train_fft, Y_train)
poly_svc = svm.SVC(kernel = 'poly', degree = 3, C = 1.0).fit(X_train_feat_density_and_X_train_fft, Y_train)


h = .02  # step size in the mesh
 
# create a mesh to plot in

X_train_min, X_train_max = X_train_feat_density.min() - 1, X_train_feat_density.max() + 1
Y_train_min, Y_train_max = X_train_fft.min() - 1, X_train_fft.max() + 1
X_train, yy = np.meshgrid(np.arange(X_train_min, X_train_max, h),
	                     np.arange(Y_train_min, Y_train_max, h))
# title for the plots
titles = ['SVC with linear kernel',
	   'LinearSVC (linear kernel)',
	    'SVC with RBF kernel',
	    'SVC with polynomial (degree 3) kernel']
 
 
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
	 # Plot the decision boundarY_train. For that, we will assign a color to each
	 # point in the mesh [X_train_min, X_train_max]X_train[Y_train_min, Y_train_max].
	 plt.subplot(2, 2, i + 1)
	 plt.subplots_adjust(wspace=0.4, hspace=0.4)
 
	 Z = clf.predict(np.c_[X_train.ravel(), yy.ravel()])
 
	 # Put the result into a color plot
	 Z = Z.reshape(X_train.shape)
	 plt.contourf(X_train, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
 
	 # Plot also the training points
	 plt.scatter(X_train_feat_density, X_train_fft, c=Y_train, cmap=plt.cm.coolwarm)
	 plt.xlabel('X_train_feat_density')
	 plt.ylabel('X_train_fft')
	 plt.xlim(X_train.min(), X_train.max())
	 plt.ylim(yy.min(), yy.max())
	 plt.xticks(())
	 plt.yticks(())
	 plt.title(titles[i])
 
plt.show()

# '''
# SVM#2 X_train_feat_density and pts_diff_avg
# '''


# C = 1.0
# svc = svm.SVC(kernel = 'linear', C=1.0).fit(X_train_feat_density_and_pts_diff_avg, Y_train)
# lin_svc = svm.LinearSVC(C=1.0).fit(X_train_feat_density_and_pts_diff_avg, Y_train)
# rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.7, C=1.0).fit(X_train_feat_density_and_pts_diff_avg, Y_train)
# poly_svc = svm.SVC(kernel = 'poly', degree = 3, C = 1.0).fit(X_train_feat_density_and_pts_diff_avg, Y_train)


# h = .02  # step size in the mesh
 
# # create a mesh to plot in

# X_train_min, X_train_max = X_train_feat_density.min() - 1, X_train_feat_density.max() + 1
# Y_train_min, Y_train_max = X_train_feat_pts_diff.min() - 1, X_train_feat_pts_diff.max() + 1
# X_train, yy = np.meshgrid(np.arange(X_train_min, X_train_max, h),
# 	                     np.arange(Y_train_min, Y_train_max, h))
# # title for the plots
# titles = ['SVC with linear kernel',
# 	   'LinearSVC (linear kernel)',
# 	    'SVC with RBF kernel',
# 	    'SVC with polynomial (degree 3) kernel']
 
 
# for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
# 	 # Plot the decision boundarY_train. For that, we will assign a color to each
# 	 # point in the mesh [X_train_min, X_train_max]X_train[Y_train_min, Y_train_max].
# 	 plt.subplot(2, 2, i + 1)
# 	 plt.subplots_adjust(wspace=0.4, hspace=0.4)
 
# 	 Z = clf.predict(np.c_[X_train.ravel(), yy.ravel()])
 
# 	 # Put the result into a color plot
# 	 Z = Z.reshape(X_train.shape)
# 	 plt.contourf(X_train, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
 
# 	 # Plot also the training points
# 	 plt.scatter(X_train_feat_density, X_train_feat_pts_diff, c=Y_train, cmap=plt.cm.coolwarm)
# 	 plt.xlabel('X_train_feat_density')
# 	 plt.ylabel('X_train_feat_pts_diff')
# 	 plt.xlim(X_train.min(), X_train.max())
# 	 plt.ylim(yy.min(), yy.max())
# 	 plt.xticks(())
# 	 plt.yticks(())
# 	 plt.title(titles[i])
 
# plt.show()

# '''
# SVM#3 X_train_feat_stdv and pts_diff_avg
# '''


# C = 1.0
# svc = svm.SVC(kernel = 'linear', C=1.0).fit(X_train_feat_stdv_and_pts_diff_avg, Y_train)
# lin_svc = svm.LinearSVC(C=1.0).fit(X_train_feat_stdv_and_pts_diff_avg, Y_train)
# rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.7, C=1.0).fit(X_train_feat_stdv_and_pts_diff_avg, Y_train)
# poly_svc = svm.SVC(kernel = 'poly', degree = 3, C = 1.0).fit(X_train_feat_stdv_and_pts_diff_avg, Y_train)


# h = .02  # step size in the mesh
 
# # create a mesh to plot in

# X_train_min, X_train_max = X_train_feat_stdv.min() - 1, X_train_feat_stdv.max() + 1
# Y_train_min, Y_train_max = X_train_feat_pts_diff.min() - 1, X_train_feat_pts_diff.max() + 1
# X_train, yy = np.meshgrid(np.arange(X_train_min, X_train_max, h),
# 	                     np.arange(Y_train_min, Y_train_max, h))
# # title for the plots
# titles = ['SVC with linear kernel',
# 	   'LinearSVC (linear kernel)',
# 	    'SVC with RBF kernel',
# 	    'SVC with polynomial (degree 3) kernel']
 
 
# for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
# 	 # Plot the decision boundarY_train. For that, we will assign a color to each
# 	 # point in the mesh [X_train_min, X_train_max]X_train[Y_train_min, Y_train_max].
# 	 plt.subplot(2, 2, i + 1)
# 	 plt.subplots_adjust(wspace=0.4, hspace=0.4)
 
# 	 Z = clf.predict(np.c_[X_train.ravel(), yy.ravel()])
 
# 	 # Put the result into a color plot
# 	 Z = Z.reshape(X_train.shape)
# 	 plt.contourf(X_train, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
 
# 	 # Plot also the training points
# 	 plt.scatter(X_train_feat_stdv, X_train_feat_pts_diff, c=Y_train, cmap=plt.cm.coolwarm)
# 	 plt.xlabel('X_train_feat_stdv')
# 	 plt.ylabel('X_train_feat_pts_diff')
# 	 plt.xlim(X_train.min(), X_train.max())
# 	 plt.ylim(yy.min(), yy.max())
# 	 plt.xticks(())
# 	 plt.yticks(())
# 	 plt.title(titles[i])

# plt.show()

'''
SVM # 4

'''

# print(np.shape(X_test))
# print(np.shape(Y_test))

Y_pred = lin_svc.predict(X_test_feat_density_and_X_test_fft)

print("Accuracy", metrics.accuracy_score(Y_test, Y_pred))
'''

Here is the code where I tried to actuallY_train plot a sin function to
the waves however, I could not figure out how to do that

'''

# def test(X_train, a, b):
#     return a * np.sin(b * X_train)

# param, param_cov = curve_fit(test, time1, volt1)
# ans = (param[0]*(np.sin(param[1]*time1)))

# #plt.scatter(volt1, time1)
# plt.plot(volt1, ans, '--', color = 'red')
# plt.show()

# print(params)



#how to build those tuples?