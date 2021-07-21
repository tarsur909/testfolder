# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Skeleton script for:
# Machine Learning - Supervised learning, a classification task
# Exercise based on Chapter 3 of Hands-On Machine Learning with
# sci-kit and tensorflow by Aurelien Geron.


# Include these packages to support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Other packages that we need to import for use in our script
import numpy as np
import os

# to make this script's output stable across runs, we need to initialize the random number generator to the same starting seed each time
np.random.seed(42)

# To save nice figures
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures that our code generates
def save_fig(fig_id, tight_layout=True):
    path = os.path.join("ML_" + fig_id + ".png") #PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

####################
# STEP 1: OBTAIN THE DATA
####################

# Now, we need to download a dataset to work with
# The sklearn package has a function that can download the data for us:

from sklearn.datasets import fetch_openml

# TO DO: use the fetch_openml function to download the dataset
#        and load it into our workspace with the variable name
#        'mnist'.  The fetch_openml function takes one argument,
#        the name of the dataset we want to download. The name
#        of the dataset to download is 'mnist_784'.
mnist = fetch_openml('mnist_784')


# TO DO: Our mnist variable is like a dictionary. It contains multiple
# fields that are named and have different types of data in them. If we want
# to look at just the field names, we can use the
# "keys()" function by entering our variable name followed immediately by
# ".keys()"  (but without the quotes around it):
# ...

# TO DO: let's organize our dataset into two variables, one a matrix that contains
#        all the descriptions of the handwritten numbers (in the "data" key of the
#        dictionary) and one a vector that contains the labels for each of those
#        numbers, the label being the actual number that was written (in the "target"
#        key of the dictionary).
#        Let's name the matrix of handwriting data "X" and the vector of labels "y":
X = mnist['data']
y = mnist['target']


# TO DO: Now, we can check the size of our dataset by using the shape function (this is
#        one of the few times where we need to call it without adding parenthesis at the
#        the end). We can call this function on our dataset by adding ".shape" to the
#        variable name (note: if nothing appears in the output terminal, wrap this
#        line with a print command):
# X...
print(X.shape, y.shape)

# Now we see that our matrix has 70,000 rows, meaning it describes 70,000
# handwritten numbers. The matrix has 728 columns, so each of the 70,000
# handwritten numbers in our database is described by 728 data points. These points
# define a 28 pixel x 28 pixel square. There is one data point for each pixel in the
# square, which gives the shade of gray in that square (anything from white to black
# and all the shades of gray in-between). If we were to graph a figure using the data
# for one of those handwritten numbers, we would get a 28x28 sized square that depicts
# a single hand-written number.

####################
# STEP 2: EXPLORE THE DATA
####################

# TO DO: Import the plotting package matplotlib. Uncomment the following
#        line to import matplotlib and to have a short way to refer
#        to its subpackage pyplot:
import matplotlib.pyplot as plt

# TO DO: Let's look at one of the numbers described in our matrix. First, define a new
#        variable called 'some_digit' that is equal to one of the rows of data in matrix X:
some_digit = X[0]
# TO DO: Now, we need to reshape that row into a square, the 28x28 square that can
#        represent the picture of that handwritten number. We can do this in two
#        steps. First take a subset of our X variable, where we define the row
#        and column indices we want in our subset. We want to use the value of
#        some_digit as our row. And we want all columns (all indices in the row) to
#        be included in our data subset, so keep in mind that the argument we provide
#        for the column argument of X will not be a single number.
#        After slicing the X matrix to get the data from a single row, we can
#        call the reshape function of that subset, which takes two arguments,
#        the row and column dimensions to reshape our variable into, and save
#        this reshaped subset as a new variable, some_digit_image:
some_digit_image = some_digit.reshape(28,28)

# TO DO: Next, we need to use pyplot's "imshow" function to show our image. The arguments
#        it takes are the 28x28 matrix we just made to describe the image, a colormap that
#        tells pyplot which colors are represented by which values found in the matrix, and
#        a method for handling values in the matrix that don't exactly match the values
#        defined in the color map (ex: if the color map only says 0.1 = almost black and
#        0.5 = medium gray, pyplot needs to know what to do for the value 0.3). Uncomment
#        the following lines to show a picture of the handwritten number:
plt.imshow(some_digit_image)
plt.axis('off')
# What number is shown in the image of some_digit?

# Edit the file name if you wish
import matplotlib
# TO DO: What if we want to look at other numbers in our dataset? It's more efficient
#        to write a function that repeats the steps we just did:
#        Given a row of the matrix, the function should reshape it into a 28x28 square,
#        plot it in an image using the imshow function, and then turn the axis off so
#        that the resulting figure looks like an image and not a graph. Try defining a
#        function called 'plot_digit' that accomplishes this task, where 'data' is the
#        name given the incoming argument (the row of the matrix with 728 data points).
#        Remember to indent the substatements of the function by four spaces:
def plot_digit(data):
    data_image = data.reshape(28,28)
    plt.imshow(data_image)
    plt.axis('off')
    
#    substatements go here...
#    substatements go here...
#    substatements go here...

    
# TO DO: Now, let's test out our new function on the handwritten number in row
#        3600 of our X matrix. We should be able to call it as follows:
plot_digit(X[36000])

# What number do you see when you call that function? Is the number hard to read?
# What number is the computer likely to think it is? If you are unsure what number
# it is, can you think of an easy way to find out?

# TO DO: Sometimes we may want to view a whole bunch of numbers at once. For that,
#        we can define a different function that shows many numbers together. The
#        function is already written for you here, just uncomment each line. As you
#        uncomment each line, take a look at it to see if you can figure out what
#        that line of code is doing:
def plot_digits(instances, images_per_row=10, **options):
     size = 28
     images_per_row = min(len(instances), images_per_row)
     images = [instance.reshape(size,size) for instance in instances]
     n_rows = (len(instances) - 1) // images_per_row + 1
     row_images = []
     n_empty = n_rows * images_per_row - len(instances)
     images.append(np.zeros((size, size * n_empty)))
     for row in range(n_rows):#
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
     image = np.concatenate(row_images, axis=0)
     plt.imshow(image, cmap = matplotlib.cm.binary, **options)
     plt.axis("off")


# TO DO: Now, we can make use of this new function to show many handwritten
#        numbers at a time. Uncomment the following lines to show many numbers:
plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
plt.show()
plt.savefig("exampleimages.png")



####################
# STEP 2: SEPARATE TRAINING AND TEST DATA
####################

# TO DO: We need to separate both our number data matrix (X) and
#        the vector of labels that contains the true identity of
#        each number (y). We have 70,000 numbers total, so lets
#        use 60,000 numbers for training and 10,000 for test.
#        Use your knowledge of slicing arrays to select the first
#        60,000 numbers for training and the last 10,000 for testing:
X_train = X[0: 60000]
X_test =  X[60000:70000]
y_train = y[0: 60000]
y_test =  y[60000:70000]


# Next, let's shuffle the data in case there is an order to the
# numbers in our matrix. Machine learning algorithms may be thrown
# off if a lot of the same number are presented in a row or if the
# order of numbers presented follows a specific pattern. Therefore,
# we will randomly rearrange the numbers in our test data set to
# disrupt any patterns in the presentation of the numbers.

# TO DO: Uncomment the following code to load in a new module and
#        create a vector that will list the row indices of our
#        test matrix in random order. Then we can use that vector
#        to rearrange the test matrix:
import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


####################
# STEP 3: TRAIN THE NETWORK
####################

# We'll first start with a simpler task of having the network
# only differentiate between 5s and everything else. THis task,
# in which every number will be separated into one of two
# categories, is called a binary classification task.

# We need to copy the rows from our data matrix that correspond
# to the number 5. However, the only way to know which rows
# represent 5s is to consult our label vector. We can create a
# vector of row indices that correspond to 5s from our label
# vector using the commands below. Notice that we have to do
# this both for the training vector and for the test vector.

# TO DO: Uncomment the code below to create vectors that
#        list whether each entry in the training and test sets
#        is a 5 or is not a 5:
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

# Now, we are ready to train our network using the training
# data. The algorithm we want to use, SGDClassifier, must
# be imported from the sklearn module.

# TO DO: Uncomment the code below to import the algorithm:
from sklearn.linear_model import SGDClassifier

# TO DO: Now, we need to call the SGDClassifier. This function
#        requires two named arguments, max_iter and random_state.
#        The max_iter argument sets the maximum number of
#        iterations to perform while learning the task (it is
#        a coincidence that this number should be set to 5). The
#        random_state argument lets us specify a random seed
#        to use to initialize our network. Lets use 42. Set the
#        output of the function to 'sgd_clf'. This represents
#        our learning algorithm object.
#
sgd_clf = SGDClassifier(max_iter = 5, random_state = 42)

# TO DO: Next, we need to provide our learning algorithm with the
#        training data and its corresponding labels. Because we
#        created a new vector called y_train_5, our labels for this
#        task are no longer the actual number represented by the
#        training data, but are instead all set to 0 for numbers
#        that are not 5 and to 1 for numbers that are 5.  Call the
#        fit() method of the sgd_clf object with two arguments, the
#        training data set and the new label vector (with 0 for non-5
#        and 1 for 5).
# sgd_clf ...

sgd_clf.fit(X_train, y_train_5)
# TO DO: Now let's see what our trained algorithm thinks that strange
#        looking number is. Uncomment and run the following code to
#        find out:
some_digit = X_train[43679]
prediction = sgd_clf.predict([some_digit])

# TO DO: Write a print statement that prints out the
#        predicted value and the actual value:
print(prediction, y_train_5[43679])

# TO DO: Let's get the predictions for all training data:
y_train_pred = sgd_clf.predict(X_train)

####################
# STEP 5: EVALUATE THE NETWORK PERFORMANCE
####################

# To DO: Now we want to measure the performance of the training algorithm
#        taking into account all of the training data. We can find out the
#        exact performance of the algorithm because we know the real identities
#        of the number samples and we know what the algorithm came up with
#        for each sample. We need to import the function 'cross_val_score'
#        from sklearn.model_selection and then call that function with the
#        arguments 'sgd_clf' (the learning algorithm object), our training
#        dataset (X_train), our label vector that labels whether each training row
#        is a 5 or not a 5 (called y_train_5, don't use the other vector y_train
#        for this exercise), and two named arguments: cv=3, scoring="accuracy"
from sklearn.model_selection import cross_val_score
performance = cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = 'accuracy')
print("cross_val_score = ", performance)

# Before we go any further, let's discuss the performance
# of the model and then see how we can better measure the performance.
#
# Important concepts here are:
# - precision: accuracy of the positive predictions of the model. When
#                the model says a number is a "5", how often is it right?
# - recall: sensitivity of the model. How many "5"s in the dataset does
#                it correctly detect?
#
# When we calculated the cross_val_score above, it looks really good at
# over 90% accuracy. But that is only because the 5s make up about 10% of
# the dataset. If it just categorizes every single number as "not-5", then
# it already achieves 90% accuracy without doing anything useful.

# TODO: A better measure of performance is the confusion matrix. To produce a 
#       confusion matrix that will give us a better indication of the model
#       performance, import the confusion_matrix function from sklearn.metrics
#       and then call that function with two arguments, the target classifications
#        (y_train_5) and the predicted classifications (y_train_pred):
from sklearn.metrics import confusion_matrix
newperformance = confusion_matrix(y_train_5, y_train_pred)
print(newperformance)

#
# We can also calculate values for precision and recall directly, by importing and
# using the precision_score and recall_score functions from sklearn.metrics (and 
# passing in the same two arguments as when we called the confusion_matrix function)
from sklearn.metrics import recall_score, precision_score
precision = precision_score(y_train_5, y_train_pred)
recall = recall_score(y_train_5, y_train_pred)
print("precision = ", precision, ", recall = ", recall)

# The model performance is a balance between precision and recall. If we alter the
# sensitivity of the model, we can increase precision but at the expense of recall,
# and vice versa. Sometimes people combine these two performance measures into a 
# single performance score called the F1 score, which will only be high (desirable)
# if both precision and recall are sufficiently strong. To calculate the F1 score,
# import and use the f1_score function from sklearn.metrics:
from sklearn.metrics import f1_score
f1performance = f1_score(y_train_5, y_train_pred)
print("F1 score = ", f1performance)



####################
# BONUS TASK
# MULTICLASS CLASSIFICATION
####################

# Now we can perform a more intuitive machine learning task on our data. Our dataset
# contains ten different digits, so it makes sense to allow for the classification of
# each digit into a category for its own identity. There are several ways to approach this
# task. We can either use an algorithm that inherently supports multiclassification. Or
# we can use a combination of binary classifiers to sort our data into multiple classes.
#
# If we use a combination of binary classifiers, we can go about it in two different ways.
# We can either train one binary classifier on each target label in the dataset, and subject
# each datum to every binary classifier; the one with the highest score "wins." This is
# called "one-versus-all" or OvA. Or we can train binary classifiers for every possible
# pair of digits (ex: 0 or 1? 0 or 2? 1 or 2? 0 or 3? ... and so on). For the MNIST dataset,
# this means 45 different binary "competitions." Each digit has to be run through all 45
# of these competitions, and whichever digit classifier wins the most of them is then considered
# the predicted class of that digit. This technique is refered to as "one-versus-one" or OvO.
# Which technique you use depends on the type of binary classifier you are using and the
# properties and size of your dataset.
#
# Let's compare the performance of two different approaches to multiclassification:
# - using a set of binary classifiers in an OvA configuration
# - using a multiclassifier
#
# TODO: Use the SDGclassifier again. This time, we will be working with y_train labels
#        (not y_train_5 labels). Scroll or "find" many lines back in this script where
#        you used the "fit" and "predict" functions of the sgd_clf class. Use them again
#        here with the same data (X_train) and the appropriate set of labels (y_train).
sgd_clf.fit(X_train, y_train)
sgd_clf.predict(X_train)
#
# TODO: Now see how this algorithm does on that same digit we used before, except now the
#        algorithm is performing a multiclassification task. Uncomment the next two lines:
some_digit_scores = sgd_clf.decision_function([some_digit])
print("scores for some_digit=", some_digit_scores)
#
# Notice that scores are returned for each potential label of the digit, from 0 to 9.
# Which score is the highest?
#
# TODO: To programmatically arrive at the highest-scoring category, uncomment these lines
#        of code. Try to understand what each line does as you uncomment it:
n = np.argmax(some_digit_scores)
prediction = sgd_clf.classes_[n]

# Now, let's try another approach, a multiclassifier algorithm called Random Forest.

# TODO: call training function (fit) and a prediction function (prediction) for the
#        random forest classifier (forest_clf), passing in the same arguments as when
#        we used the previous algorithm:
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier()
forest_clf.fit(X_train, y_train)
forest_prediction = forest_clf.predict(X_train)

# This function called directly gave us the predicted label for our "some_digit". But
# what if we want to see how it rated all the possible options? If we want to see the
# scores assigned to each potential label, we can run this line of code to get them:
possible_scores = forest_clf.predict_proba([some_digit])
print(possible_scores)

# So we saw how each of these algorithms assigned our some_digit, and the relative scores
# each algorithm gave all 10 possible classifications for some_digit. What if we want
# to measure the performance of the algorithms on our test data?
#
# TODO: Compute the cross_val_score on our revised sgd_clf and on our forest_clf algorithms,
#         supplying the same types of arguments as we did previously when we computed the
#        cross_val_score for our 5/not-5 binary classification. Except that instead of y_train_5
#        labels, we will this time supply y_train for our labels argument to the function. You
#        may need to scroll or "find" back up in the code to see how you called cross_val_score
#        previously:
sgd_score = cross_val_score(sgd_clf, X_train, y_train, cv = 3, scoring = 'accuracy')
forest_score = cross_val_score(forest_clf, X_train, y_train, cv = 3, scoring = 'accuracy')

# You are welcome to compare other performance metrics for these two algorithms. Which algorithm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05, 0.1],
    'learning_rate': ['constant','adaptive'],
}

mlp = MLPClassifier()
random = RandomizedSearchCV(mlp, param_grid, n_jobs=-1, cv=3)
random.fit(X_train, y_train)
mlp = MLPClassifier(**random.best_params_)
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))


base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False)

 
