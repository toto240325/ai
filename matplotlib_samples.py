'''
@author: zhaosong
'''
# Import pyplot module and alias it as plt. This can avoid repeatly call pyplot.
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


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

def draw_vertical_columnbar_line():
    
    # The x-axis maximum number.
    axis_x = 16
    # The y-axis maximum number.
    axis_y = 10**2
    # Set x-axis and y-axis minimum and maximum number.
    plt.axis([0, axis_x, 0, axis_y])
    
    x_value = 1
    
    x_delta = 1
    
    y_tuple = (0.8, 0.6, 0.5, 0.3, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9)
    
    for y_percent in y_tuple:
        
        color_value = 'green'
        
        if y_percent < 0.6:
            
            color_value = 'red'
        
        plt.axvline(x=x_value, ymin=0, ymax=y_percent, color=color_value, label=y_percent)
        
        x_value = x_value + x_delta
        
    # Add legend label.
    plt.legend()
    plt.show()

def draw_vertical_columnbar_line_with_stem_method():
    
    # The x-axis maximum number.
    axis_x_max = 16
    
    axis_y_max = 100
    
    x_value = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    y_value = [80, 60, 50, 30, 60, 70, 80, 50, 60, 70, 80, 90]
    
    plt.xlim(0, axis_x_max)
    
    plt.ylim(0, axis_y_max)
    
    plt.stem(x_value, y_value)
    plt.show()


def draw_horizontal_line():
    
    # The x-axis maximum number.
    axis_x = 100

    # The y-axis maximum number.
    axis_y = 10
    
    # Set x-axis and y-axis minimum and maximum number.
    plt.axis([0, axis_x, 2, axis_y])
    
    # plot a green horizontal line, the start point is (0,5), the end point is (0.8*axis_x, 5).
    plt.axhline(5, 0, 0.8, color='green', label='plot a green horizontal line')
    
    # plot a red horizontal line, the start point is (0.1*axis_x,6), the end point is (0.9*axis_x, 6).
    plt.axhline(6, 0.1, 0.9, color='red', label='plot a red horizontal line')
    # Add the label provided in the above axvline() method
    plt.legend()
    plt.show()  


def draw_line():
    # List to hold x values.
    x_number_values = [1, 2, 3, 4, 5]
    # List to hold y values.
    y_number_values = [1, 4, 9, 16, 25]
    # Plot the number in the list and set the line thickness.
    plt.plot(x_number_values, y_number_values, linewidth=3)
    # Set the line chart title and the text font size.
    plt.title("Square Numbers", fontsize=19)
    # Set x axis label.
    plt.xlabel("Number Value", fontsize=10)
    # Set y axis label.
    plt.ylabel("Square of Number", fontsize=10)
    # Set the x, y axis tick marks text size.
    plt.tick_params(axis='both', labelsize=9)
    # Display the plot in the matplotlib's viewer.
    plt.show()


# draw a line of given slope and intercept with y axis
def draw_things(slope=1,intercept=3):
    # line equation : y = slope*x + intercept
    x1 = 1
    x2 = 5
    x0 = 0
    y0 = slope * x0 + intercept
    y1 = slope * x1 + intercept
    y2 = slope * x2 + intercept
    
    # List to hold x values.
    x_number_values = [x0,x1,x2]
    # List to hold y values.
    y_number_values = [y0,y1,y2]
    # Plot the number in the list and set the line thickness.
    plt.plot(x_number_values, y_number_values, linewidth=3)
   
    # plotting a line
    xpoints = np.array([0, 6])
    ypoints = np.array([0, 5])
    plt.plot(xpoints, ypoints)

    # plotting points
    xpoints = np.array([1, 8])
    ypoints = np.array([3, 10])
    plt.plot(xpoints, ypoints, 'o')

    # # plotting based on y values only (x will range from 0, 1, 2, 3, ...)
    # ypoints = np.array([3, 8, 1, 10, 5, 7])
    # plt.plot(ypoints)

    # # plot points with a specific marker
    # # a lot of different markers : https://www.w3schools.com/python/matplotlib_markers.asp
    # ypoints = np.array([13, 1.8, 12, 9])
    # plt.plot(ypoints, marker = '*')

   

    ypoints = np.array([13, 1.8, 12, 9])

    # # a market|line|column indication
    # # possible line values : '-', '--', ':', '-.'
    # # colours : r: red, g: green, b: blue, c: cyan, m: magenta, y: yellow, k: black, w: white
    # plt.plot(ypoints, 'o:r')

    # # market size : 
    # plt.plot(ypoints, marker = 'o', ms = 20)

    # # market edge colour : 
    # plt.plot(ypoints,'o:b', ms = 20, mec = 'r')
    
    # linestyle : 
    plt.plot(ypoints, linestyle = 'dotted')
    
    # linewidth : 

    ypoints2 = ypoints + 1
    plt.plot(ypoints2, linewidth = 10)
    



    # Set the line chart title and the text font size.
    plt.title(f"simple line with slope={slope} and intercept={intercept}", fontsize=19)
    # Set x axis label.
    plt.xlabel("X Values", fontsize=10)
    # Set y axis label.
    plt.ylabel("Y Values", fontsize=10)
    # Set the x, y axis tick marks text size.
    plt.tick_params(axis='both', labelsize=9)
    
    # plt.xlim(0, 10)

    # Display the plot in the matplotlib's viewer.
    plt.show()


def dynamic_plot():
    x = np.linspace(0, 6*np.pi, 100)
    y = np.sin(x)

    # You probably won't need this if you're embedding things in a tkinter plot...
    # ion : interactive mode on
    plt.ion()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma

    for phase in np.linspace(0, 10*np.pi, 500):
        line1.set_ydata(np.sin(x + phase))
        fig.canvas.draw()
        fig.canvas.flush_events()


if __name__ == '__main__':
    # draw_point()
    # draw_multiple_points()
    # draw_point_with_auto_generate_values()
    # save_colorful_point_to_picture()
    # draw_vertical_line()
    # draw_vertical_columnbar_line()
    # draw_vertical_columnbar_line_with_stem_method()
    # draw_horizontal_line()
    # draw_line()
    # draw_things()
    dynamic_plot()

    
    