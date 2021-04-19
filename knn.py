#!/usr/bin/env python3

import numpy;

def euclideanDistance(trainingVector, testVector):
    # print("euclideanDistance trainingVector length:",len(trainingVector[0]));
    # print("euclideanDistance testVector length:",len(testVector));

    return numpy.linalg.norm(trainingVector[::,1:86] - testVector[1:86]);

def getEuclideanDistanceForPoint(trainingPoint, test):
    # print("OK")

    # print("Test", test[0]);
    # print("Train", trainingPoint[0]);
    # print(test);
    # import sys; sys.exit(1);
    # Return the id, distance from the test point, and income for each training point
    return [trainingPoint[0], euclideanDistance(trainingPoint, test), trainingPoint[86]]

def knn(train, testPoint, n):

    # print(len(train))

    # For each point in the training data get the euclideanDistance from the test point
    pointNeighborDistance = numpy.array(
        [
            getEuclideanDistanceForPoint(trainingPoint, testPoint)
            for trainingPoint in train
        ]
    );

    # Sort by the euclideanDistance value
    pointNeighborDistance = pointNeighborDistance[pointNeighborDistance[:,1].argsort()];

    # If test data has income, return the income
    # if(len(testPoint == 87)):
    #     pointNeighborDistance = [testPoint[:1][0], pointNeighborDistance, testPoint[-1:][0]];
    # else:
    pointNeighborDistance = [testPoint[:1][0], pointNeighborDistance];

    pointNeighborDistance = numpy.asarray(pointNeighborDistance, dtype=object);

    return pointNeighborDistance;


def getNearestNeighborsForTestingData(train, test, k):
    
    # print("Training", len(train))
    # print("Testing", len(test))


    return numpy.array(
        [knn(train, testPoint, k) for testPoint in test]
    );

def fourFoldCrossValidation(k, allTrainingData):

    trainingSubsetLength = int(numpy.floor(len(allTrainingData)/4));
    # print("training length", trainingLength);

    trainingSubsets = [];

    for i in range(0, 4):

        bottom = i * trainingSubsetLength;
        top = (i+1) * trainingSubsetLength;

        trainingSubsets.append(allTrainingData[bottom:top])

    # Convert to numpy array
    trainingSubsets = numpy.asarray(trainingSubsets);

    nearestNeighbors = [];

    for i in range(0, 4):
        training = numpy.delete(trainingSubsets, i, axis=1);
        validation = trainingSubsets[i];
        print("Training ", len(training[0]));
        print(len(trainingSubsets[i]));

        nearestNeighbors.append(getNearestNeighborsForTestingData(training, validation, k));

    nearestNeighbors = numpy.asarray(nearestNeighbors);

    print(nearestNeighbors);

    # print(numpy.array(nearestNeighbors)[0][1]);

def getAccuracyForNearestNeighborData(neighborData):
    # [[testID, [[trainID, distance, income]...], income?]]

    print(neighborData);

    # numpy.array(
        # [knn(train, testPoint, k) for testPoint in testData]
        # [for neighbor>]
    # )?


def importData(filename):
    my_data = numpy.genfromtxt(filename, delimiter=',', skip_header=1, skip_footer=1);
    print("Data Loaded.");
    print("Dimensions:\t", len(my_data[0]));
    print("Data points:\t", len(my_data));
    print("=====================");
    return my_data

def main():
    # Remove income column
    trainingData = importData("train.csv");

    testingData = importData("test_pub.csv");
    
    testKs = [1,3,5,7,9,99,999,8000];

    k = 5;

    # nearestNeighbors = getNearestNeighborsForTestingData(trainingData, testingData[:5], k);

    # getAccuracyForNearestNeighborData(nearestNeighbors);

    # print(nearestNeighbors);

    fourFoldCrossValidation(5, trainingData);


if __name__ == "__main__":
    main()
