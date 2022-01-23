import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pandas as pd
from math import log2


# calculate the entryp of this df
def entropy(df,kind="Species"):
    kinds = df[kind].unique()
    total = len(df)
    nb_per_kind = dict()
    entropy_per_kind = dict()
    for k in kinds:
        nb_per_kind[k] = len(df[(df[kind] == k)])
        entropy_per_kind[k] = - nb_per_kind[k]/total * log2(nb_per_kind[k]/total) 
    return sum(entropy_per_kind.values())


# calculate information gain if split with given query
def information_gain_criterion(df, entropy_whole_set, df_query, kind):
    entropy_subsets = dict()
    nb_subsets = dict()

    sub_df = df.query(df_query)
    entropy_subsets[0] = entropy(sub_df,kind)
    nb_subsets[0] = len(sub_df)
    # print(f'Entropy for {df_query} : {entropy_subsets[0]}')

    df_query = f"not ({df_query})"
    sub_df = df.query(df_query)
    entropy_subsets[1] = entropy(sub_df,kind)
    nb_subsets[1] = len(sub_df)
    # print(f'Entropy for {df_query} : {entropy_subsets[1]}')

    return entropy_whole_set - (entropy_subsets[0] * nb_subsets[0] + entropy_subsets[1] * nb_subsets[1] ) / len(df)


df = pd.read_csv("data_bugs.csv")

entropy_whole_set = entropy(df,"Species")
print(f'Entropy whole set : {entropy_whole_set}')

df_query = 'Color == "Green"'
ig  = information_gain_criterion(df, entropy_whole_set, df_query, "Species")
print(f'information gain for {df_query} : {ig}')

df_query = 'Color == "Brown"'
ig  = information_gain_criterion(df, entropy_whole_set, df_query, "Species")
print(f'information gain for {df_query} : {ig}')

df_query = 'Color == "Blue"'
ig  = information_gain_criterion(df, entropy_whole_set, df_query, "Species")
print(f'information gain for {df_query} : {ig}')

df_query = 'Length < 17'
ig  = information_gain_criterion(df, entropy_whole_set, df_query, "Species")
print(f'information gain for {df_query} : {ig}')

df_query = 'Length < 20'
ig  = information_gain_criterion(df, entropy_whole_set, df_query, "Species")
print(f'information gain for {df_query} : {ig}')


exit()


# let's split by color and see entropy of resulting sets :
colors = df["Color"].unique()
entropy_subsets = dict()
for g in colors:
    sub_df = df[(df["Color"]==g)]
    entropy_subsets[g] = entropy(sub_df,"Species")
    print(f'Entropy for {g} : {entropy_subsets[g]}')
info_gain_per_color = entropy_whole_set - sum(entropy_subsets.values())/len(entropy_subsets)
print(f'information gain if split by color = {info_gain_per_color}')

# let's split by length17 and see entropy of resulting sets :
entropy_subsets = dict()

sub_df = df[(df["Length"] < 17)]
entropy_subsets["lessThan17"] = entropy(sub_df,"Species")
print(f'Entropy for less than 17 : {entropy_subsets["lessThan17"]}')

sub_df = df[(df["Length"] >= 17)]
entropy_subsets["moreThan17"] = entropy(sub_df,"Species")
print(f'Entropy for more than 17 : {entropy_subsets["moreThan17"]}')

info_gain_per_length17 = entropy_whole_set - sum(entropy_subsets.values())/len(entropy_subsets)
print(f'information gain if split by lengh17 = {info_gain_per_length17}')


# let's split by length20 and see entropy of resulting sets :
entropy_subsets = dict()

sub_df = df[(df["Length"] < 20)]
entropy_subsets["lessThan20"] = entropy(sub_df,"Species")
print(f'Entropy for less than 20 : {entropy_subsets["lessThan20"]}')

sub_df = df[(df["Length"] >= 20)]
entropy_subsets["moreThan20"] = entropy(sub_df,"Species")
print(f'Entropy for more than 20 : {entropy_subsets["moreThan20"]}')

info_gain_per_length20 = entropy_whole_set - sum(entropy_subsets.values())/len(entropy_subsets)
print(f'information gain if split by lengh20 = {info_gain_per_length20}')




sub_df = df[(df["Color"] == "Green")]
entropy_subsets["lessThan20"] = entropy(sub_df,"Species")
print(f'Entropy for less than 20 : {entropy_subsets["lessThan20"]}')

sub_df = df[(df["Color"] >= 20)]
entropy_subsets["moreThan20"] = entropy(sub_df,"Species")
print(f'Entropy for more than 20 : {entropy_subsets["moreThan20"]}')

info_gain_per_Color20 = entropy_whole_set - sum(entropy_subsets.values())/len(entropy_subsets)
print(f'information gain if split by lengh20 = {info_gain_per_Color20}')


exit()


nb_lobugs = len(df[(df["Species"] =="Lobug")])
nb_mobugs = len(df[(df["Species"] =="Mobug")])
nb_bugs = nb_lobugs + nb_mobugs
entropy_all_bugs = - nb_lobugs/nb_bugs * log2(nb_lobugs/nb_bugs) - nb_mobugs/nb_bugs * log2(nb_mobugs/nb_bugs)





exit()
df = pd.read_csv("data_bugs.csv")

# calculate entropy all bugs
nb_lobugs = len(df[(df["Species"] =="Lobug")])
nb_mobugs = len(df[(df["Species"] =="Mobug")])

nb_bugs = nb_lobugs + nb_mobugs

entropy_all_bugs = - nb_lobugs/nb_bugs * log2(nb_lobugs/nb_bugs) - nb_mobugs/nb_bugs * log2(nb_mobugs/nb_bugs)

# calculate entropy if we split by color

colors = df.Color.unique()

nb_bugs_per_color = dict()
entropy_per_color = dict()
for c in colors:
    nb_bugs_per_color[c] = len(df[(df["Color"] ==c)])
    entropy_per_color[c] = - nb_bugs_per_color[c]/nb_bugs * log2(nb_bugs_per_color[c]/nb_bugs) 

# print(nb_bugs_per_color)
# print(entropy_per_color)



X = df["Length"].values
Y = [ (1 if col == "Blue" else (2 if col == "Brown" else 3)) for col in df["Color"]]
# Y = [ (1 if row["Color"] == "Blue" else (2 if row["Color"] == "Brown" else 3)) for row in df.iterrows()]
C = [ ("r" if spec == "Lobug" else "b") for spec in df["Species"]]
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


