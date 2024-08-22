"""
This program finds the best value of K in KMeans algorithm using Silhouette Coefficient for 'housing.csv' dataset. The range of K values to analyze is provided as a command line parameter.
Syntax: python assignment.py <number> <number>

For example, to search best K between 3 and 6 the command line input should be:
python assignment.py 3 6
"""

# importing the libraries

"""  DO NOT MODIFY  """
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
"""  DO NOT MODIFY  """

def find_best_kmeans(data, min_k, max_k):

    """  write from here  """

    best_k = k_min
    best_score = -1
    print(data)
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k

        
    """        End        """


"""  DO NOT MODIFY  """
if __name__ == '__main__':

    """
    ALERT: * * * No changes are allowed in this section  * * *
    """
 
    if len(sys.argv) == 2:
        print("Usage: python assignment.py <number> <number>")
        sys.exit(1)

    input_data_one = sys.argv[1].strip()
    input_data_two = sys.argv[2].strip()
    print(input_data_one)
    """  Call to function that will perform the computation. """
    if input_data_one.isdigit() and input_data_two.isdigit():

        min_k = int(input_data_one)
        max_k = int(input_data_two)
        if min_k>=2 or max_k>min_k:
            data =pd.read_csv("/Users/mac/code/Programming_assignment_week_6/housing.csv")
            print(data)
            print(find_best_kmeans(data, min_k, max_k)) 
        else:
           print("Invalid input")
    else:
        print("Invalid input")
 
    
    """ End to call """