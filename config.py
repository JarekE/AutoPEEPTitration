# Define configs here

# System

model = "SimpleNN"      #SimpleNN, Philip, kNN, SVM

epoch_number = (3000 * 10) # number must be  X % 10 = 0

learning_rate = 0.001

# Cross-Validation: Choose between 7 different splits (Training/Test/Validation)
# With all 7 splits every data-set is used for testing (exactly 1 time)
split = 3 # Choose a number between 1-7


# IMPORTANT: Only test your network with the same split used for training (otherwise an error will occur)
# (if False, the test with the correct split is executed automatically)
test = False

# Calculates the data with 50:50 optimal peep step and not optimal peep step values by
# adding copies of the optimal peep step values at the end of the vector
# (Idea: Since the optimal-peep-step-data-points are only 10-12% of the data, the network tends to learn weights to calculate
# only not-optimal-peep-steps, which leads to a accuracy of up ~90%)
equal_size = False

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
# compare ML models with each other
compare = False
# Do not use the peep to train the neural network (we recommend this setting to be True)
no_peep = True
# Save the trainings report
report = True
