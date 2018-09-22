# inspired by Siraj Raval's clip: https://www.youtube.com/watch?v=T5pRlIbr6gg&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU

# import the GaussianNB algorithm from scikit-learn's Naive Bayes
from sklearn.naive_bayes import GaussianNB

# data to work with
x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], 
     [160, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], 
     [159, 55, 37], [171, 75, 42], [181, 85, 43]]

y = ['male', 'female', 'female', 'female', 'male', 'male', 
     'male', 'female', 'male', 'female', 'male']

# create a shorter name for the algorithm
gnb = GaussianNB()

# fit the data
gnb = gnb.fit(x, y)

# make the prediction
prediction = gnb.predict([[190, 70, 43]])

# output the result
print(prediction)
