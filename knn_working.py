# Example of kNN implemented from Scratch in Python
from __future__ import print_function
import csv
import random
import math
import operator


def loadDataset(filename, trainingSet=[]):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	#print(len(dataset),range(len(dataset)))		
	
	for x in range(len(dataset)):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])	        
	            trainingSet.append(dataset[x])
	        
def loadDataset1(filename, testSet=[]):
	lines1 = csv.reader(open(filename, "r"))
	dataset1 = list(lines1)
	print(len(dataset1),range(len(dataset1)))		
	
	for x in range(len(dataset1)):
	        for y in range(4):
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
	# prepare data
	trainingSet=[]
	testSet=[]	
	split = 0.99
	#loadDataset('iris.data','iris_test.data', split, trainingSet, testSet)
	loadDataset('iris.data', trainingSet)
	
	loadDataset1('iris_test.data', testSet)
	
	print ('Train set: ',repr(len(trainingSet)))	
	print ('Test set: ', repr(len(testSet)))
	print ('Test: ',testSet)
	
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)		
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()