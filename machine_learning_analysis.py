import matplotlib.pyplot as plt

# Creates General plot line chart for Data
def plot_line_chart(x, y, label='Sample Data', xlabel='xlabel', ylabel='ylabel', title='Insert Title', figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.plot(x, y, label=label)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    
    plt.legend()
    plt.show()
