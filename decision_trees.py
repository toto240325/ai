# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

def timenow():
    return time.time()

class Elapsed_time:
    def __init__(self):
        self.mytimes = []
        self.mytimes.append(timenow())

    def elapsed_time(self, point):
        self.mytimes.append(timenow())
        i = len(self.mytimes)
        diff_secs = self.mytimes[i-1] - self.mytimes[i-2]
        print(f'delta{i-1} {point} : {round(diff_secs,2)}')

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


elaps = Elapsed_time()

plt.ion()


# Read the data.
data = np.asarray(pd.read_csv('data_decision_trees.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

colorY = [ 'b' if label == 0 else 'r' for label in y]

fig1 = plt.figure(figsize=(4,4))
ax1 = fig1.add_subplot(111)
fig1.add_subplot(ax1)
move_figure(fig1,20,20)

# # fig, ax = plt.subplots()
# mngr = plt.get_current_fig_manager()
# # to put it into the upper left corner for example:
# mngr.window.setGeometry(50,100,640, 545)

f, ax = plt.subplots()
move_figure(f, 500, 500)


ax1.scatter(X[:,0], X[:,1], color = colorY)

elaps.elapsed_time("preparation")

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 3)
# model = DecisionTreeClassifier(max_depth = 7)
#model = DecisionTreeClassifier()

# TODO: Fit the model.
model.fit(X,y)

# to visualize the decision tree :
# https://mljar.com/blog/visualize-decision-tree/
text_representation = tree.export_text(model)
print(text_representation)

# fig2 = plt.figure(figsize=(5,5))
# # ax1 = fig1.add_subplot(111)
# # fig1.add_subplot(ax1)
# move_figure(fig2,500,500)

_ = tree.plot_tree(model, 
                    fontsize=8,
                   filled=True)


elaps.elapsed_time("model.fit")

# TODO: Make predictions. Store them in the variable y_pred.

#new_points = [ [0.2, 0.8], [0.5, 0.4] ]

#new_points = np.random.randint(10, size=(5, 2))
new_points = np.random.random(size=(8, 2))

y_pred = model.predict(new_points)
elaps.elapsed_time("model.predict")
color_pred = [ "b" if y_p == 0 else "r" for y_p in y_pred]

# print(f'new_points: {new_points}')

for i in range(len(new_points)):
    new_p = new_points[i]
    y_p = y_pred[i]
    # print(f'new_p : {new_p}')
    x0 = new_p[0]

    x1 = new_p[1]
    ax1.scatter([x0],[x1], s=100, color="b" if y_p == 0 else "r")

# TODO: Calculate the accuracy and assign it to the variable acc.

new_points = [ [x[0],x[1]] for x in X]
y_pred = model.predict(new_points)

acc = accuracy_score(y, y_pred)
print(f"accuracy : {acc}")

fig1.canvas.get_tk_widget().focus_force()
plt.show()
d = model.get_depth()
print(f'depth : {d}')

d2 = model.tree_.max_depth
print(f'depth2 : {d2}')

nc = model.tree_.node_count
print(f'node count : {nc}')

params = model.get_params()
print(f'params : {params}')


input("press a key")

exit()
