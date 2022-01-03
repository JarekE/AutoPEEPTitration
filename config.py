# Define configs here

#System

model = "SVM" #RNN oder SimpleNN oder Philip oder LSTM oder SVM

#User

# how many breathes per peep step (10,20,40)
length = 20
# only usable for the SimpleNN in the moment
epoch_number = 10000

# calculate the gradient and use it
grad = True

# plot the data with the classes
print = False