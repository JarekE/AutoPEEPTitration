# Define configs here

# System

model = "SimpleNN"      #SimpleNN, Philip, kNN, SVM

train_ratio = 0.8       # must add up to 1.0
validation_ratio = 0.1
test_ratio = 0.1

epoch_number = (3000 * 10) # number must be  X % 10 = 0

learning_rate = 0.01

# IMPORTANT: Activate to test saved model "model.pth" --> no training of network (-> Attention for data)
# If False, one also test the trained model (one just calculated)
# Only test models, if random state is set (42) for training the saved model, otherwise data-split issues will occur
random_state = False #42
test = False

#----------------------------------------------------#

# User

# how many breathes per peep step (10,20,40)
length = 20
# calculate the gradient and use it
grad = True
# print data plots of all input peep-data-sets
print = False
# plot curve of loss (--> overfitting)
loss_plot = False
# save classification report as text file
report = False
# compare ML models with each other
compare = False