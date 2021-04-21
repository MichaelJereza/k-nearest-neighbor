#!/usr/bin/env python3

import numpy;
from scipy import stats;

def euclideanDistance(trainingVector, testVector):
    # print("euclideanDistance trainingVector length:",len(trainingVector));
    # print("euclideanDistance testVector length:",len(testVector));

    return numpy.linalg.norm(trainingVector[1:86] - testVector[1:86]);

def getEuclideanDistanceForPoint(trainingPoint, test):
    # print("OK")

    # print("Test", test[0]);
    # print("Train", trainingPoint[0]);
    # print(test);
    # import sys; sys.exit(1);
    # Return the id, distance from the test point, and income for each training point
    return [trainingPoint[0], euclideanDistance(trainingPoint, test), trainingPoint[86]]

knnRepeats = 0;

def knn(train, testPoint, k):

    neighbors = numpy.array(
        [
            getEuclideanDistanceForPoint(trainingPoint, testPoint)
            for trainingPoint in train
        ],
        dtype=object
    );

    # Get the first k neighbors
    neighbors = neighbors[neighbors[:,1].argsort()];

    neighbors = neighbors[:k];

    global knnRepeats;
    knnRepeats += 1;
    global totalTests;
    print("Getting",len(neighbors),"nearest neighbors for test point", knnRepeats, "/", totalTests, "\t\t", end="\r");

    # If test data has income, return the income
    if(len(testPoint == 87)):
        pointNeighborDistance = [
            testPoint[:1][0],  # test point ID
            neighbors, # neighbors [trainingID, distance, income]
            testPoint[-1:][0] # income
        ];
    else:
        pointNeighborDistance = [
            testPoint[:1][0], 
            neighbors
        ];

    pointNeighborDistance = numpy.asarray(pointNeighborDistance, dtype=object);

    return pointNeighborDistance;


def getNearestNeighborsForTestingData(train, test, k):
    
    # print("Training", len(train))
    # print("Testing", len(test))
    # print();
    # print(k);

    global knnRepeats;
    knnRepeats = 0;
    global totalTests;
    totalTests = len(test);

    return numpy.array(
        [knn(train, testPoint, k) for testPoint in test]
    );

def fourFoldCrossValidation(k, allTrainingData):

    trainingSubsetLength = int(len(allTrainingData)/4);
    # print("training length", trainingLength);

    # if(k > trainingSubsetLength):

    trainingSubsets = [];

    for i in range(0, 4):

        bottom = i * trainingSubsetLength;
        top = (i+1) * trainingSubsetLength;

        if(i==3):
            trainingSubsets.append(allTrainingData[bottom:]);
        else:
            trainingSubsets.append(allTrainingData[bottom:top]);

    # Convert to numpy array
    trainingSubsets = numpy.asarray(trainingSubsets, dtype=object);

    nearestNeighbors = [];

    for i in range(0, 4):
        training = numpy.delete(trainingSubsets, i, 0);
        validation = trainingSubsets[i];

        # print("Training on section lengths", len(training[0]), len(training[1]), len(training[2]))
        
        training = numpy.concatenate(training);

        # print("Getting neighbors for fold", i+1);
        # print("Training Set Length", len(training));
        # print("Validation Set Length", len(validation));
        # print("First trainingPoint ID:", training[0][0]);
        # print("Last trainingPoint ID:", training[-1][0]);
        # print("====================================");

        nearestNeighbors.append(getNearestNeighborsForTestingData(training, validation, k));


    nearestNeighbors = numpy.asarray(nearestNeighbors, dtype=object);

    print("\nRESULTS");
    print("Folds", len(nearestNeighbors));
    print("Fold Size", len(nearestNeighbors[0]));
    # print("Attributes per Test", len(nearestNeighbors[0][0]));

    return nearestNeighbors;

def checkCorrectClassification(neighbor, classification):
    if(neighbor[2] == classification):
        return 1;
    else:
        return 0;

def getTestAccuracy(test):
    # print(len(test[1]))
    # print("Checking", test[0], "for income matching", test[2])
    # print(test[-2])
    accuratePredictions = numpy.array([
        checkCorrectClassification(neighbor, test[2]) for neighbor in test[1]
    ])

    return numpy.sum(accuratePredictions)/len(test[1])

def getAveragesForFold(fold):
    foldTestAverages = numpy.array([getTestAccuracy(test) for test in fold])
    # print(foldTestAverages);
    return numpy.average(foldTestAverages);

def getAccuracyForNearestNeighborData(k, neighborData):
    # [[testID, [[trainID, distance, income]...], income?]]

    print("Processing data...");

    foldAverages = numpy.array([getAveragesForFold(fold) for fold in neighborData])

    kAverage = numpy.average(foldAverages);
    kVariance = numpy.var(foldAverages);

    print("ACCURACY")
    print("Mean:", kAverage);
    print("Variance", kVariance);

    # print(foldAverages);
    return (k, kAverage, kVariance);    


def importData(filename):
    my_data = numpy.genfromtxt(filename, delimiter=',', skip_header=1, skip_footer=1);
    print("Data Loaded.");
    print("Dimensions:\t", len(my_data[0]));
    print("Data points:\t", len(my_data));
    print("=====================");
    return my_data

def testKValueAccuracy(k, trainingData):
    print("\nChecking accuracy for k =",k);
    nn = fourFoldCrossValidation(k, trainingData);
    return getAccuracyForNearestNeighborData(k, nn);

def determineBestK(kValues, trainingData):
    accuracyForKs = numpy.array([testKValueAccuracy(k, trainingData) for k in kValues], dtype=[("k", int), ("mean", float), ("variance", float)]);

    bestVars = numpy.sort(accuracyForKs, order="variance")[::-1]

    bestMeans = numpy.sort(accuracyForKs, order="mean")[::-1]
        
    # print(numpy.array(bestMeans[:,:1] - bestMeans[:,:1]))
    displayAccuracies(bestMeans, bestVars);

    # kValues = [];

    score = 0;
    bestFittingK = -1;

    for mean in range(len(bestMeans)):
        for variance in range(len(bestVars)):
            if(bestMeans[mean][0]==bestVars[variance][0]):

                # Get the index of the mean and variance which is where it ranked in the sorted array
                thisScore = mean+1 + variance+1;
                
                # If the mean and variance are better than the current pick replace it
                if(thisScore < score or score == 0):
                    bestFittingK = bestMeans[mean][0];
                    score = thisScore;
                


    # import sys; sys.exit(1);

    return bestFittingK;

def displayAccuracies(means, vars):
    print("\n\t==Mean==\t==Variance==")
    for i in range(len(means)):
        print(i+1,"\t", "k=",means[i][0], end="\t\t\t")
        print( "k=",vars[i][0]);
    
    print();

def getTestPredictions(nearestNeighborsForPoint):
    # print((nearestNeighborsForPoint[1])[:,2]);
    bestGuess = stats.mode((nearestNeighborsForPoint[1])[:,2])[0][0];
    # print(bestGuess);
    return (int(nearestNeighborsForPoint[0]), int(bestGuess));

def main():
    # Remove income column
    trainingData = importData("train.csv");

    testingData = importData("test_pub.csv");
    
    testKs = [1,3,5,7,9,99,999,8000];

    # nearestNeighbors = getNearestNeighborsForTestingData(trainingData, testingData[:5], k);

    # getAccuracyForNearestNeighborData(nearestNeighbors);

    # print(nearestNeighbors);
    bestK = determineBestK(testKs, trainingData);
  
    testDataNeighbors = getNearestNeighborsForTestingData(trainingData, testingData, bestK);

    print();

    print(bestK);

    predictions = numpy.array(
        [
            getTestPredictions(neighbors) for neighbors in testDataNeighbors
        ]
    )

    print(predictions);

    with open("guesses.csv", "wb") as f:
        f.write(b'id,income\n')
        numpy.savetxt(f, predictions.astype(int), fmt='%i', delimiter=",")
        f.close()

if __name__ == "__main__":
    main()
