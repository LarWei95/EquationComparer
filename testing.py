'''
Created on 03.11.2021

@author: larsw
'''
import numpy as np

from eqcomparer import EquationComparer
from pprint import pprint
import matplotlib.pyplot as plt

def main():
    value_count = 10000
    
    index1 = np.linspace(-1, 1, value_count)
    index2 = np.logspace(-3, 3, value_count)
    index3 = np.linspace(4, -4, value_count)
    
    truth1 = np.sinh(index1)
    truth2 = np.sqrt(index2)
    truth3 = np.tanh(index3)
    truth4 = truth1 * truth2
    truth5 = truth1 / truth2
    
    index_dict = {
            "Index1" : index1,
            "Index2" : index2,
            "Index3" : index3
        }
    truth_dict = {
            "Truth1" : truth1,
            "Truth2" : truth2,
            "Truth3" : truth3,
            "Truth4" : truth4,
            "Truth5" : truth5
        }
    eq_dict = {
            "Eq1" : truth1,
            "Eq2" : truth2,
            "Eq3" : truth3,
            "Eq4" : truth4,
            "Eq5" : truth5,
            "Eq1M" : truth1 + 0.05,
            "Eq2M" : np.log(truth2),
            "Eq3M" : np.sin(truth3),
            "Eq4M" : truth1 + truth2,
            "Eq5M" : truth2 + truth3 / truth1
        }
    
    equation_comp = EquationComparer()
    
    truth_dict.update(EquationComparer.calculate_differentials(index_dict, truth_dict, 1))
    eq_dict.update(EquationComparer.calculate_differentials(index_dict, eq_dict, 1))
    
    eq_comparations = equation_comp.compare(truth_dict, eq_dict)
    rd_eq_comparations = equation_comp.reduce_comparations(eq_comparations)
    pprint(rd_eq_comparations.transpose())
    
    assignment = equation_comp.assign_equations_to_truths(rd_eq_comparations)
    print(assignment)
    
    for eq_key in assignment.index:
        truth_key = assignment["Truth"][eq_key]
        score = assignment["Score"][eq_key]
        
        eq_values = eq_dict[eq_key]
        truth_values = truth_dict[truth_key]
        
        title = "{:s} --> {:s} [{:.4f}]".format(eq_key, truth_key, score)
        plt.title(title)
        
        plt.plot(eq_values, label="Equation")
        plt.plot(truth_values, label="Truth")
        plt.legend()
        plt.grid(True)
        plt.show()
    
if __name__ == '__main__':
    main()