'''
@author: zhaosong
'''
# Import pyplot module and alias it as plt. This can avoid repeatly call pyplot.
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Draw a point based on the x, y axis value.
def draw_point():
    # Draw a point at the location (3, 9) with size 1000
    plt.scatter(3, 9, s=1000)
    plt.scatter(4, 16, s=500)
    # Set chart title.
    plt.title("Square Numbers", fontsize=19)
    # Set x axis label.
    plt.xlabel("Number", fontsize=10)
    # Set y axis label.
    plt.ylabel("Square of Number", fontsize=10)
    # Set size of tick labels.
    plt.tick_params(axis='both', which='major', labelsize=9)
    # Display the plot in the matplotlib's viewer.
    plt.axis([1, 5, 0, 20])
    plt.show()

# Draw multiple points.
def draw_multiple_points():
    # x axis value list.
    x_number_list = [1, 4, 9, 16, 25]
    # y axis value list.
    y_number_list = [1, 2, 3, 4, 5]
    # Draw point based on above x, y axis values.
    plt.scatter(x_number_list, y_number_list, s=10)
    # Set chart title.
    plt.title("Extract Number Root ")
    # Set x, y label text.
    plt.xlabel("Number")
    plt.ylabel("Extract Root of Number")
    plt.show()

# Draw a serial of points which x, y axis value is calculated by range function.
def draw_point_with_auto_generate_values():
    # Set the x axis number max value.
    x_number_max = 100
    # Auto generate x number value list by range function.
    x_number_list = list(range(1, x_number_max))
    # Y axis value is assigned by x**3
    y_number_list = [x**3 for x in x_number_list]
    # Draw points based on above x, y value list and remove the point black outline. And the point color is green.
    plt.scatter(x_number_list, y_number_list, s=10, edgecolors='none', c='green')
    # Set x, y axis min and max number.
    plt.axis([0, x_number_max, 0, x_number_max**3])
    plt.show()

# Draw color point with color map and save result to a picture.
def save_colorful_point_to_picture():
    x_number_max = 100
    x_number_list = list(range(1, x_number_max))
    y_number_list = [x**2 for x in x_number_list]
    # Draw points which remove the point black outline. And the point color is changed with color map.
    plt.scatter(x_number_list, y_number_list, s=10, edgecolors='none', c=y_number_list, cmap=plt.cm.Reds)
    # Set x, y axis minimum and maximum number.
    plt.axis([0, x_number_max, 0, x_number_max**2])
    plt.show()
    # Save the plot result into a picture this time.
    plt.savefig('test_plot.png', bbox_inches='tight')


def draw_vertical_line():
    # The x-axis maximum number.
    axis_x = 10
    
    # The y-axis maximum number.
    axis_y = 5
    
    # Set x-axis and y-axis minimum and maximum number.
    plt.axis([2, axis_x, 0, axis_y])
    
    # plot a green vertical line, the start point is (5, 0), the end point is (5, 0.8*axis_y).
    plt.axvline(5, 0, 0.8, color='green', label='plot a green vertical line')
    
    # plot a red vertical line, the start point is (8.5, 0.1*axis_y), the end point is (8.5, 0.6*axis_y).
    plt.axvline(8.5, 0.1, 0.6, color='red', label='plot a red vertical line')
    # Add the label provided in the above axvline() method
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # draw_point()
    # draw_multiple_points()
    # draw_point_with_auto_generate_values()
    # save_colorful_point_to_picture()
    draw_vertical_line()
    
    