import matplotlib.pyplot as plt

def plot_image(data, actual=None, predicted=None, ax=None):
    if ax == None:
        ax = plt.gca()
    ax.imshow(data, cmap=plt.cm.binary)
    if actual != None:
        plt.annotate('Actual:' + str(actual), xy=(16,16), xytext=(26,1), 
                     horizontalalignment='right', verticalalignment='top')
    if predicted != None:
        plt.annotate('Predicted:' + str(predicted), xy=(16,16), xytext=(26,3), 
                     horizontalalignment='right', verticalalignment='top')
    
