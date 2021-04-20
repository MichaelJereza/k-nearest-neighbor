#!/usr/bin/env python3

import numpy;

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

def knn(train, testPoint, n):

    # print(len(train))

    # For each point in the training data get the euclideanDistance from the test point
    neighbors = numpy.array(
        [
            getEuclideanDistanceForPoint(trainingPoint, testPoint)
            for trainingPoint in train
        ],
        dtype=object
    );


    # trainingPointIds = train[::,0];

    # trainnigPointIncomes = train[::,-1];

    # print("=++++++++++++++++++=")
    
    # print("ID")
    # print(len(trainingPointIds));
    # print(trainingPointIds[-4]);

    # print("euclidean Distance")
    # print(len(pointNeighborDistance));
    # print(pointNeighborDistance[-4]);

    # print("Income")
    # print(len(trainnigPointIncomes));
    # print(trainnigPointIncomes[-4]);

    # print(train)
    # print(len(train))
    # print(len(pointNeighborDistance))

    # neighbors = numpy.hstack((trainingPointIds, pointNeighborDistance, trainnigPointIncomes));


    # print("=-------------------=")

    # print(len(neighbors));

    # Sort neighbors by the euclideanDistance value
    neighbors = neighbors[neighbors[:,1].argsort()];

    # print("=-------------------=")
    # print("Nearest Neighbor out of", len(neighbors));

    # print("ID")
    # print(neighbors[0][0]);

    # print("euclidean Distance")
    # print(neighbors[0][1]);

    # print("Income")
    # print(neighbors[0][2]);

    # print("=zzzzzzzzzzzzzzzzzzz=")
    
    # print("ID")
    # print(neighbors[-4][0]);

    global knnRepeats;
    knnRepeats += 1;
    global totalTests;
    print("Getting",n,"nearest neighbors for test point", knnRepeats, "/", totalTests, end="\r");

    # print("euclidean Distance")
    # print(len(neighbors[1]));
    # print(neighbors[-4][1]);

    # print("Income")
    # print(len(neighbors[2]));
    # print(neighbors[-4][2]);

    # print("=++++++++++++++++++=")

    # If test data has income, return the income
    if(len(testPoint == 87)):
        pointNeighborDistance = [
            testPoint[:1][0],  # test point ID
            neighbors, # neighbors [trainingID, distance, income]
            testPoint[-1:][0]
        ];
    else:
        pointNeighborDistance = [
            testPoint[:1][0], 
            neighbors
        ];

    pointNeighborDistance = numpy.asarray(pointNeighborDistance, dtype=object);


    # print("Attached ID and income?")

    # print(len(pointNeighborDistance))
    # print(pointNeighborDistance)


    return pointNeighborDistance;


def getNearestNeighborsForTestingData(train, test, k):
    
    print("Training", len(train))
    print("Testing", len(test))
    print();
    global knnRepeats;
    knnRepeats = 0;
    global totalTests;
    totalTests = len(test);

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

        if(i==3):
            trainingSubsets.append(allTrainingData[bottom:]);
        else:
            trainingSubsets.append(allTrainingData[bottom:top]);

    # Convert to numpy array
    trainingSubsets = numpy.asarray(trainingSubsets, dtype=object);

    nearestNeighbors = [];

    for i in range(0, 4):
        training = numpy.delete(trainingSubsets, i);
        validation = trainingSubsets[i];

        
        training = numpy.concatenate(training);

        print("Getting neighbors for fold", i+1);
        print("Training on section length", len(training[0]), len(training[1]), len(training[2]))
        print("Training Set Length", len(training));
        print("Validation Set Length", len(validation));
        print("First trainingPoint ID:", training[0][0]);
        print("Last trainingPoint ID:", training[-1][0]);

        nearestNeighbors.append(getNearestNeighborsForTestingData(training, validation, k));
        print("====================================");


    nearestNeighbors = numpy.asarray(nearestNeighbors, dtype=object);

    print("Results");
    print("Folds", len(nearestNeighbors));
    print("Fold Size", len(nearestNeighbors[0]));
    print("Attributes per Test", len(nearestNeighbors[0][0]));

    return nearestNeighbors;
    # For each fold
        # For each test point
            # For each nearest neighbors
                # if neighbor.income = test.income
                    # +1
                # else
                    # 0
            # accuracy = score/neighbors
        # average accuracy per fold
    # average accuracy over all folds

    # print(nearestNeighbors[0][0])

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
