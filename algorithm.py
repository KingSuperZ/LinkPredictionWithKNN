from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import networkx as nx

"""Predicts interactions in graph data, also known as Link Prediction.

Takes an edge list and gets all combinations of numbers into one edge list
and it makes another list that respectively shows which values interact or not
based on the original data. The data is designed so all sets of numbers that are
2 units away from each other are interacting."""

def EdgetoExtraAdjacency(edgeList):
    """Converts an edge list into an adjacency list

    ExtraAdjacency is used to show how we included the inverse of each interaction to give the program more data to work with"""

    keys = [] # Creates a list to store the keys which are every unique number in the dataset
    values = [] # Creates a list to store the values for each key based on the original data

    # This block of code uses X to makes the lists keys and values as follows
    # keys:   [1, 3, 2, 4, 3, 5, -1, 1, 0, 2, -2, 0]
    # values: [3, 1, 4, 2, 5, 3, 1, -1, 2, 0, 0, -2]
    # The keys are all of the item from X in order and the values is the same thing
    # except the numbers from each list in X are reversed.
    for i in range(len(edgeList)):
        keys.append(edgeList[i][0]) # Adds the x coordinates to the keys list from before
        values.append(edgeList[i][1])
        keys.append(edgeList[i][1]) # Adds the y coordinates to the keys list from before
        values.append(edgeList[i][0])

    # This block of code take the values from the list keys to create a dictonary
    # with the list being the keys and the values being empty lists
    # adjList: {1: [], 3: [], 2: [], 4: [], 5: [], -1: [], 0: [], -2: []}
    adjList = {}
    for i in range(len(keys)):
        adjList[keys[i]] = []

    # This block of code uses the values list and adds to the dictionary posDict by
    # adding the values to each key depending on the interactions listed in X
    # adjList: {1: [3, -1], 3: [1, 5], 2: [4, 0], 4: [2], 5: [3], -1: [1], 0: [2, -2], -2: [0]}
    for i in range(len(values)):
        adjList[keys[i]].append(values[i])
    return adjList

def negativeSampling(adjList):
  # This block of code uses the list posDict to subtract all of the included value
  # from all of the existing values so that every combination of keys and values will
  # be included except for the ones in posDict
  # negDict: {1: [0, 1, 2, 4, 5, -2], 3: [0, 2, 3, 4, -1, -2], 2: [1, 2, 3, 5, -1, -2], 4: [0, 1, 3, 4, 5, -2, -1], 5: [0, 1, 2, 4, 5, -2, -1], -1: [0, 2, 3, 4, 5, -2, -1], 0: [0, 1, 3, 4, 5, -1], -2: [1, 2, 3, 4, 5, -2, -1]}
  negDict = adjList.copy()
  dict_keys = list(adjList.keys())
  for i in negDict:
    negDict[dict_keys[i]] = list(set(dict_keys)-set(negDict[dict_keys[i]]))
  return negDict

def AdjacencytoEdge(adjList):
    """Converts an adjacency list to an edge list

    Unlike the function EdgetoExtraAdjacency it doesn't add extra values for efficiency
    """
    edgeList = []
    dict_keys = adjList.keys()
    for key in adjList:
      for value in adjList[key]:
        tempList = [key,value]
        edgeList.append(tempList)
    return edgeList

# Start of the project code
data = [[1,3],[2,4],[3,5],[-1,1],[0,2],[-2,0]] # Original Data (Edge List)

# Contains the original data in an adjacency list including the inverse interactions
# Example: Before: [[1,2],[3,4]], After: [[1,2],[3,4],[2,1],[4,3]]
posDict = EdgetoExtraAdjacency(data)
print(posDict)

# Contains an adjacency list with all of the values that don't interact
negDict = negativeSampling(posDict)
print(negDict)

# This code block converts the dictionary posDict to an edge list that contains
# the different interactions
posList = AdjacencytoEdge(posDict)

# This code block converts the dictionary negDict to a list of lists that contain
# the values that don't interact
negList = AdjacencytoEdge(negDict)

# This code block makes a list the contains a mix of the two lists posList and negList
X = []
for i in posList:
    X.append(i)
for i in negList:
    X.append(i)
print(X)

# This code block makes a list containing zeros that allign with the list X to
# show whether a certain set a of values interact or not.
zeros = [0]*len(negList)
ones = [1]*len(posList)
y = []
for i in ones:
    y.append(i)
for i in zeros:
    y.append(i)
print(y)

# This block of code contains the data needed for the scikit learn algorithm
Xarray = np.array(X)
yarray = np.array(y)
Xtrain, Xtest, ytrain, ytest = train_test_split(Xarray, yarray, test_size=0.2)
# This is where the algorithm is being implemented
alg = KNeighborsClassifier(n_neighbors=1)
alg.fit(Xtrain,ytrain)
# This is where the predicted values as a result of the algorithm is stored
ypred = alg.predict(Xtest)
# This is simply formatting the data so it is distinct to the reader
# Green: Doesn't Interact, Red: Does Interact
# Dots: Training Data, Crosses: Testing Data
color1 = np.where(ytrain == 0, "green","red")
color2 = np.where(ypred == 0, "green","red")
xcord = Xtrain[:,0]
ycord = Xtrain[:,1]
xcord2 = Xtest[:,0] # Stores the x coordinates
ycord2 = Xtest[:,1]
plt.scatter(xcord,ycord, c = color1)
plt.scatter(xcord2,ycord2, c = color2, marker = "x", s = 50)
plt.axis("equal")
plt.grid()
print(accuracy_score(ytest, ypred))
