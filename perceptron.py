import matplotlib.pyplot as plt

# test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
# correct_outputs = [False, False, False, True]
# outputs = []

# a = zip(test_inputs, correct_outputs)
# print(a)
# for test_input, correct_output in zip(test_inputs, correct_outputs):
#     linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias


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


