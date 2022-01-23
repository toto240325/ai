import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pandas as pd

def display_line(w1,w2,b):
    # calculate 2 random points of the line
    X = [minX,maxX]
    Y = []
    for x in X:
        y = (-w1 * x - b)/w2
        Y.append(y)

    # plot that line
    line1, = ax.plot(X, Y, 'r-') # Returns a tuple of line objects, thus the comma
    return line1

def update_line(line1, w1, w2, b):
    # calculate 2 random points of the line
    X = [minX,maxX]
    Y = []
    for x in X:
        y = (-w1 * x - b)/w2
        Y.append(y)

    # plot that line

    line1.set_xdata(X)
    line1.set_ydata(Y)
    fig.canvas.draw()
    fig.canvas.flush_events()


def new_coef(w1, w2, b, alpha, p, categ):

    # if point p classified correctly ?
    linearcomb = w1 * p[0] + w2 * p[1] + b
    categ_OK = (linearcomb > 0 and categ == 1) or (linearcomb < 0 and categ == 0) or (linearcomb == 0)
    # it's classified correctly don't do nothing, otherwise update the coefs
    if not categ_OK:
        mult = 1 if (linearcomb < 0) else -1
        new_w1 = w1 + p[0] * alpha * mult
        new_w2 = w2 + p[1] * alpha * mult
        new_b = b + alpha * mult
    else:
        new_w1 = w1
        new_w2 = w2
        new_b = b
    return new_w1, new_w2, new_b


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)


w1 = 3
w2 = 4
b = -10
alpha = 0.01
epochs = 25
delay = 0.00

#mypoints = [(0,0,1)]
#mypoints = [(10,10,0)]
# mypoints = [(1,1,1), (10,2,0)]
mypoints = [(1,1,0), (1,2,0),(1,1,0), (1.5,2.2,0), (3,3,1), (2,1,1), (5,3,1), (2.2,0.9,1)]

data = pd.read_csv("data_perceptron_trick.csv", header=None)


X = data[0].values
Y = data[1].values

mypoints = [ (row[0],row[1],row[2]) for index,row in data.iterrows()]

X = [ p[0] for p in mypoints ]
Y = [ p[1] for p in mypoints ]

r0 = data.T[0] # row 0

X2 = data[0] # column 0

frameMinX = -1
frameMaxX = 1
frameMinY = -1
frameMaxY = 1


minX = min(np.min(X) - 1, frameMinX)
maxX = max(np.max(X) + 1, frameMaxX)
minY = min(np.min(Y) -1, frameMinY)
maxY = max(np.max(Y) + 1, frameMaxY)

ax.axis([minX, maxX,minY, maxY])

for p in mypoints:
    ax.scatter(p[0],p[1],color='b' if p[2] else 'r')


line1 = display_line(w1,w2,b)   

for i in range(epochs):
    plt.title(f"Iter : {i}", fontsize=19)
    for p in mypoints:
        a = (p[0],p[1])
        w1,w2,b = new_coef(w1,w2,b,alpha, a,p[2])
    update_line(line1, w1, w2, b)
    sleep(delay)
        
input("press a key to continue")

exit()

X = [0,0,1,1]
Y = [0,1,0,1]
plt.scatter(X,Y)

# AND operator
# line passing via (0,1.1) and (1.1,0)
w1 = 2
w2 = w1
b = -2.2

# NOT operator
# line passing via (0,0.5) and (1,0.5)
w1 = 0
w2 = -2
b = 1

x1 = 0
y1 = x1 * (-w1/w2) - b/w2

x2 = 2
y2 = x2 * (-w1/w2) - b/w2
plt.plot([x1,x2],[y1,y2])
plt.show()

exit()



# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias


