#for shuffle the image 
import random
# read the csv file 
labels = 'E:/IP/Labels/labels.csv'
#create a new csv file as shuffled_labels.csv where the image address will be shuffled
shuffled_labels = 'E:/IP/Labels/shuffled_labels.csv'

labelfile = open(labels, "r")
lines = labelfile.readlines()
labelfile.close()
random.shuffle(lines)
# create shuffle csv
shufflefile = open(shuffled_labels, "w") 
shufflefile.writelines(lines)
shufflefile.close()