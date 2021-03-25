# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

def loadCsv(filename):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset


	
def loadDataset_ckd(filename, trainingSet=[]):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	#print(len(dataset),range(len(dataset)))
	for x in range(len(dataset)):
	        for y in range(15):
	            dataset[x][y] = float(dataset[x][y])	        
	            trainingSet.append(dataset[x])
				
def loadDataset_ckd1(filename, testSet=[]):
	lines1 = csv.reader(open(filename, "r"))
	dataset1 = list(lines1)
	#print(len(dataset1),range(len(dataset1)))	
	for x in range(len(dataset1)):
	        for y in range(15):
	            dataset1[x][y] = float(dataset1[x][y])	        
	            	        
	testSet.append(dataset1[x])				
				
def loadDataset_ml(filename, trainingSet=[]):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for x in range(len(dataset)):
	        for y in range(9):
	            dataset[x][y] = float(dataset[x][y])	        
	            trainingSet.append(dataset[x])				
	        
def loadDataset_hd(filename, trainingSet=[]):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)	
	for x in range(len(dataset)):
	        for y in range(12):
	            dataset[x][y] = float(dataset[x][y])	        
	            trainingSet.append(dataset[x])	
			

	
def loadDataset_ml1(filename, testSet=[]):
	lines1 = csv.reader(open(filename, "r"))
	dataset1 = list(lines1)
	#print(len(dataset1),range(len(dataset1)))
	for x in range(len(dataset1)):
	        for y in range(9):
	            dataset1[x][y] = float(dataset1[x][y])	        
	            	        
	testSet.append(dataset1[x])

def loadDataset_hd1(filename, testSet=[]):
	lines1 = csv.reader(open(filename, "r"))
	dataset1 = list(lines1)
	#print(len(dataset1),range(len(dataset1)))
	for x in range(len(dataset1)):
	        for y in range(12):
	            dataset1[x][y] = float(dataset1[x][y])	        
	            	        
	testSet.append(dataset1[x])	

	
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	#print(separated)
	for classValue, instances in separated.items():
		#print(instances)
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	#print(x,mean,stdev)

	if(x==0 and mean==0 and stdev==0):	
		x = 1
		mean = 1
		stdev = 1
		#print(x,mean,stdev)
	part2 = (2*math.pow(stdev,2))
	if(part2==0) :	
		part2 = 0.1
	#print(part2)	
	exponent = math.exp(-(math.pow(x-mean,2)/part2))
	part3 = (math.sqrt(2*math.pi) * stdev)
	if(part3==0) :	
		part3 = 0.1	
	fin = (1 / part3) * exponent
	return fin

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
    
	print ('\n~~~~~~~~~~~');
	#checking of presence of ckd disease
	# prepare data
	matched_count = 0 ;
	total_datas = 0
	trainingSet=[]
	testSet=[]
	loadDataset_ckd('dataset_ckd_train.csv', trainingSet)	
	total_datas = total_datas+int(repr(len(trainingSet)))
	loadDataset_ckd1('dataset_ckd_test.csv', testSet)	
	print ('Train set of ckd: ',repr(len(trainingSet)))	
	#print ('Train set: ', trainingSet)
	#print ('Test set: ', repr(len(testSet)))
	print ('Input for CKD disease related parameters :\n ',testSet)
	summaries = summarizeByClass(trainingSet)
	matched_count = matched_count+int(repr(len(summaries)))
	print('matches: ',repr(len(summaries)))
	# test model
	predictions = getPredictions(summaries, testSet)
	#print('> predicted=' , predictions)
	print('> disease presence =' , predictions )
	
	accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: {0}%').format(accuracy)
	#print('Accuracy: ',accuracy)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	print ('\n~~~~~~~~~~~');
	#checking of presence of diabetes disease
	trainingSet=[]	
	testSet=[]
	
	loadDataset_ml('dataset_diabetes_train.csv', trainingSet)
	total_datas = total_datas+int(repr(len(trainingSet)))
	loadDataset_ml1('dataset_diabetes_test.csv', testSet)
	print ('Train set of diabetes: ',repr(len(trainingSet)))	
	print ('Input for Diabetes disease related parameters :\n ',testSet)
	#print(trainingSet)
	#print(testSet)
	# prepare model
	summaries = summarizeByClass(trainingSet)
	#print(summaries)	
	matched_count = matched_count+int(repr(len(summaries)))
	print('matches: ',repr(len(summaries)))
	# test model
	predictions = getPredictions(summaries, testSet)
	#print('> predicted=' , predictions)
	print('> disease presence =' , predictions )
	
	accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: {0}%').format(accuracy)
	#print('Accuracy: ',accuracy)
		
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	print ('\n~~~~~~~~~~~');
	#checking of presence of heart disease	
	trainingSet=[]	
	testSet=[]
	
	loadDataset_hd('dataset_heartdisease_train.csv', trainingSet)
	total_datas = total_datas+int(repr(len(trainingSet)))
	loadDataset_hd1('dataset_heartdisease_test.csv', testSet)
	print ('Train set of heart disease: ',repr(len(trainingSet)))	
	print ('Input for heart disease related parameters :\n ',testSet)
	
	summaries = summarizeByClass(trainingSet)	
	#print(summaries)
	matched_count = matched_count+int(repr(len(summaries)))
	print('matches: ',repr(len(summaries)))
	# test model
	predictions = getPredictions(summaries, testSet)
	#print('> predicted=' , predictions)
	print('> disease presence =' , predictions )
	
	accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: {0}%').format(accuracy)
	#print('Accuracy: ',accuracy)	
	
	print('Total Datas',total_datas,'Matched Accuracy: ',matched_count)
main()