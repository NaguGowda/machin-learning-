# Example of kNN implemented from Scratch in Python
from __future__ import print_function
import csv
import random
import math
import operator
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids)/2

def iterate_k_means(data_points, centroids, total_iteration):
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    
    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point], centroids[index_centroid])
            label = assign_label_cluster(distance, data_points[index_point], centroids)
            centroids[label[0]] = compute_new_centroids(label[1], centroids[label[0]])

            if iteration == (total_iteration - 1):
                cluster_label.append(label)

    return [cluster_label, centroids]

def print_label_data(result):
    print("Result of k-Means Clustering: \n")
    for data in result[0]:
        print("data point: {}".format(data[1]))
        print("cluster number: {} \n".format(data[0]))
    print("Last centroids position: \n {}".format(result[1]))

def create_centroids():
    centroids = []
    centroids.append([5.0, 0.0])
    centroids.append([45.0, 70.0])
    centroids.append([50.0, 90.0])
    return np.array(centroids)
	
def loadDataset(filename, trainingSet=[]):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	#print(len(dataset),range(len(dataset)))
	for x in range(len(dataset)):
	        for y in range(14):
	            dataset[x][y] = float(dataset[x][y])	        
	            trainingSet.append(dataset[x])
				
def loadDataset_ml(filename, trainingSet=[]):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for x in range(len(dataset)):
	        for y in range(8):
	            dataset[x][y] = float(dataset[x][y])	        
	            trainingSet.append(dataset[x])				
	        
def loadDataset_hd(filename, trainingSet=[]):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)	
	for x in range(len(dataset)):
	        for y in range(11):
	            dataset[x][y] = float(dataset[x][y])	        
	            trainingSet.append(dataset[x])	
			
def loadDataset1(filename, testSet=[]):
	lines1 = csv.reader(open(filename, "r"))
	dataset1 = list(lines1)
	#print(len(dataset1),range(len(dataset1)))	
	for x in range(len(dataset1)):
	        for y in range(14):
	            dataset1[x][y] = float(dataset1[x][y])	        
	            	        
	testSet.append(dataset1[x])
	
def loadDataset_ml1(filename, testSet=[]):
	lines1 = csv.reader(open(filename, "r"))
	dataset1 = list(lines1)
	#print(len(dataset1),range(len(dataset1)))
	for x in range(len(dataset1)):
	        for y in range(8):
	            dataset1[x][y] = float(dataset1[x][y])	        
	            	        
	testSet.append(dataset1[x])

def loadDataset_hd1(filename, testSet=[]):
	lines1 = csv.reader(open(filename, "r"))
	dataset1 = list(lines1)
	#print(len(dataset1),range(len(dataset1)))
	for x in range(len(dataset1)):
	        for y in range(11):
	            dataset1[x][y] = float(dataset1[x][y])	        
	            	        
	testSet.append(dataset1[x])
	
	
def euclideanDistance(instance1, instance2, length):
	distance = 0	
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	#sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	print ('\n~~~~~~~~~~~');
	#checking of presence of ckd disease
	# prepare data
	trainingSet=[]
	testSet=[]
	matched_count = 0
	total_datas = 0
	loadDataset('dataset_ckd_train.csv', trainingSet)	
	loadDataset1('dataset_ckd_test.csv', testSet)		
	
	print ('Train set of ckd: ',repr(len(trainingSet)))	
	#print ('Train set: ', trainingSet)
	#print ('Test set: ', repr(len(testSet)))
	print ('Input for CKD disease related parameters :\n ',testSet)
	
	total_datas = total_datas+int(repr(len(trainingSet)))
	
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		matched_count = matched_count+int(repr(len(neighbors)))
		print('matches: ',repr(len(neighbors)))
		result = getResponse(neighbors)
		predictions.append(result)		
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
		print('> disease presence =' + repr(result) )
	accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: ' + repr(accuracy) + '%')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	print ('\n~~~~~~~~~~~');
	#checking of presence of diabetes disease
	trainingSet=[]
	testSet=[]
	loadDataset_ml('dataset_diabetes_train.csv', trainingSet)	
	print ('Train set of diabetes: ',repr(len(trainingSet)))		
	#print ('Train set: ', trainingSet)
	
	loadDataset_ml1('dataset_diabetes_test.csv', testSet)		
	#print ('Test set: ', repr(len(testSet)))
	print ('Input for Diabetes disease related parameters :\n ',testSet)
	
	total_datas = total_datas+int(repr(len(trainingSet)))
	
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		matched_count = matched_count+int(repr(len(neighbors)))
		print('matches: ',repr(len(neighbors)))
		result = getResponse(neighbors)
		predictions.append(result)		
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
		print('> disease presence =' + repr(result) )
	accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: ' + repr(accuracy) + '%')		
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	print ('\n~~~~~~~~~~~');
	#checking of presence of heart disease
	trainingSet=[]
	testSet=[]
	loadDataset_hd('dataset_heartdisease_train.csv', trainingSet)	
	print ('Train set of heart disease: ',repr(len(trainingSet)))		
	#print ('Train set: ', trainingSet)
	
	loadDataset_hd1('dataset_heartdisease_test.csv', testSet)		
	print ('Input for Heart disease related parameters : \n', repr(len(testSet)))
	print ('Test: ',testSet)	
	
	total_datas = total_datas+int(repr(len(trainingSet)))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		matched_count = matched_count+int(repr(len(neighbors)))
		print('matches: ',repr(len(neighbors)))
		result = getResponse(neighbors)
		predictions.append(result)		
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
		print('> disease presence =' + repr(result) )
	accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: ' + repr(accuracy) + '%')	
	
	print('Total Datas',total_datas,'Matched Accuracy: ',matched_count)
	

	
main()
