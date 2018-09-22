# Using Decision Tree Classifier, copied from Siraj's first clip:
# https://www.youtube.com/watch?v=T5pRlIbr6gg&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU

# Import DecisionTree from SciKit-Learn

from sklearn import tree

# Create a list that contains multiple lists
# based on the height, weight and shoe size 
# to determine the gender

x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], 
     [160, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], 
     [159, 55, 37], [171, 75, 42], [181, 85, 43]]

y = ['male', 'female', 'female', 'female', 'male', 'male', 
     'male', 'female', 'male', 'female', 'male']


# Fit the data using the DecisionTreeClassifier algorithm
clf = tree.DecisionTreeClassifier()

clf = clf.fit(x, y)

# Make predictions
prediction = clf.predict([[190, 70, 43]])

# Display the result
print(prediction)
