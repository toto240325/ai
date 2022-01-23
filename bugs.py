import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pandas as pd

data = pd.read_csv("data_bugs.csv")

X = data["Length"].values
Y = [ (1 if col == "Blue" else (2 if col == "Brown" else 3)) for col in data["Color"]]
# Y = [ (1 if row["Color"] == "Blue" else (2 if row["Color"] == "Brown" else 3)) for row in data.iterrows()]
C = [ ("r" if spec == "Lobug" else "b") for spec in data["Species"]]
plt.scatter(X,Y,color=C)

plt.show()
exit()

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


