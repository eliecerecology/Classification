#import matplotlib.pyplot as plt
#import numpy as np

#basics
'''
x = np.linspace(0,3*np.pi, 100)


def print_name(my_name):
    print "My name is: %s" %my_name



print_name ("Vikram")
y = np.random.rand(*x.shape)


plt.plot(x,y)
plt.axis([0,11,-0.6,1.6])
plt.show()

'''

#loops
'''
for i in range (0,3):
    print i
'''
#numpy loop
'''
x = np.linspace(0,3*np.pi, 100)
for i in x: #loop numpy
    print i
'''

#Basics arrange-matrices
'''
t_arrange = np.arange(0, 10, 2) #create an arrange from 0-10 by 2
t_linspace = np.linspace(0,10,5) # create sequence
z_zeros = np.zeros([4,5]) #creates a matrix 4 by 5 using zeros
z_ones = np.ones([3,3]) # creates a matrix 3 by 3
'''
#ploting
#import matplotlib.plt as plt
#import numpy as np
'''
x = np.linspace(0,2*np.pi, 100)
y = np.sin(x)
plt.plot(x,y)
plt.show()
'''
#PANDAS READING FILE
#import pandas as pd
#from sklearn import datasets #http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

#iris = datasets.load_iris() #http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
#d1 = pd.read_csv("/home/worki/Documents/CGIwork/IrisData.csv")
#d2 = pd.get_dummies(d1)
#header = d2.columns.values
#subset
'''
d1.ix[2:5,['Sepal length', 'Petal length']]
plt.plot(d1.ix[2:5,['Sepal length']], d1.ix[2:5,['Petal length']])
plt.show()
'''
###examination dataframe
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

data=pd.read_csv("/home/worki/Documents/CGIwork/IrisData.csv", names=['f1','f2','f3','f4','f5'])
'''
################################################################DATA IRIS ANALYSIS
#import desired packages
import tensorflow as tf
import numpy as np
import pandas as pd
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython
sns.set(style="white", color_codes=True)


#data=pd.read_csv("/home/ellis/Documents/CGIwork/IrisData.csv")

data = pd.read_csv("/home/ellis/Documents/CGIwork/DataNoname.csv",
                   names=['f1','f2','f3','f4','f5'])
data['Id'] = data.index

#############is it balanced?

#data.head()
#data["f5"].value_counts() 


#data.plot(kind="scatter", x="f2", y="f3")

##PCA
# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature

from pandas.tools.plotting import radviz
radviz(data.drop("Id", axis=1), "f5")

#sns.pairplot(data.drop("Id", axis=1), hue="Species", size=3) #does not work well
#data.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6)) # good one

'''
sns.boxplot(x="f5", y="f3", data=data)
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
sns.plt.show()
'''
'''
#Visualization 2 features (sepal length and Sepal width)
#map data into arrays
'''
s = np.asarray([1,0,0])
ve = np.asarray([0,1,0])
vi = np.asarray([0,0,1])

data['f6'] = data['f5'].map({'setosa': s,
                             'versicolor': ve,'virginica':vi})

###shuffle data
data = data.iloc[np.random.permutation(len(data))]


###training data
x_input = data.ix[0:105, ['f1', 'f2', 'f3', 'f4']]
temp = data['f6']
y_input = temp[0:16]
#test data
x_test=data.ix[106:149,['f1','f2','f3','f4']]
y_test=temp[106:150]


#placeholders and variables. Input has 4 features and output has 3 classes
x = tf.placeholder(tf.float32, shape=[None, 4])
y_ = tf.placeholder(tf.float32, shape=[None, 3])
#weight and bias
w=tf.Variable(tf.zeros([4,3]))
b=tf.Variable(tf.zeros([3]))

#model
#softmax function for multiple classification
y = tf.nn.softmax(tf.matmul(x, w) + b)

##loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y),
                                              reduction_indices=[1]))
#optimizer
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)


#calculate accuracy of the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#session parameter
sess = tf.InteractiveSession()
#initialising variables
init = tf.initialize_all_variables()
sess.run(init)

#number of interations
epoch=2000

for step in xrange(epoch):
    _, c=sess.run([train_step, cross_entropy], feed_dict={x: x_input, y_:[t for t in y_input.as_matrix()]})
    if step%500==0 :
        print c

#random testing at Sn.130
a=data.ix[144,['f1','f2','f3','f4']]
b=a.reshape(1,4)
largest = sess.run(tf.arg_max(y,1), feed_dict={x: b})[0]
if largest==0:
    print "flower is :Iris-setosa"
elif largest==1:
    print "flower is :Iris-versicolor"
else :
    print "flower is :Iris-virginica"

print sess.run(accuracy,feed_dict={x: x_test, y_:[t for t in y_test.as_matrix()]})

                





















