import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import numpy as np

label = ['Knn_Total','NB_Total']
no_data = [6244,7020]

def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))	
    plt.bar(index, no_data)
    plt.xlabel('Algorithm', fontsize=5)
    plt.ylabel('Data Count', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Disease Prediction')
    #plt.show()
	
    plt.show()

plot_bar_x()	
#plot_bar_x1()	