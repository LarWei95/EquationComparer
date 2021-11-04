'''
Created on 04.11.2021

@author: larsw
'''
import numpy as np
import json
import requests
from pprint import pprint
from io import StringIO
import pandas as pd

URL = "http://127.0.0.1:5000"

def ndarray_dict_to_jsonable (d):
    return {
            k : d[k].tolist()
            for k in d
        }

def jsonable_to_ndarray_dict (d):
    return {
            k : np.array(d[k])
            for k in d
        }

def differentiate_test (index_dict, truth_dict, eq_dict):
    values = {
            "indices" : ndarray_dict_to_jsonable(index_dict),
            "values" : ndarray_dict_to_jsonable(truth_dict)
        }
    
    
    r = requests.get(URL+"/differentiate?diffs=1", json=values)
    content = json.loads(r.content)
    content = jsonable_to_ndarray_dict(content)
    
    print("Differentiate:")
    pprint(content)
    
    return content

def compare_test (truth_dict, eq_dict):
    values = {
            "truths" : ndarray_dict_to_jsonable(truth_dict),
            "equations" : ndarray_dict_to_jsonable(eq_dict)
        }
    
    r = requests.get(URL+"/compare", json=values)
    print(r.status_code)
    content = r.content.decode("utf-8").replace("\r\n","\n")
    buf = StringIO(content)
    
    content = pd.read_csv(buf, sep=";", index_col=0)
    
    print("Compare:")
    print(content)
    
    return content

def assign_test (comparations):
    buf = StringIO()
    comparations.to_csv(buf, sep=";")
    values = {
            "comparations" : buf.getvalue()
        }
    
    r = requests.get(URL+"/assign", json=values)
    print(r.status_code)
    content = r.content.decode("utf-8").replace("\r\n","\n")
    
    buf = StringIO(content)
    
    content = pd.read_csv(buf, sep=";", index_col=0)
    
    print("Assign:")
    print(content)
    
    return content
    
def direct_assign_test (truth_dict, eq_dict):
    values = {
            "truths" : ndarray_dict_to_jsonable(truth_dict),
            "equations" : ndarray_dict_to_jsonable(eq_dict)
        }
    
    r = requests.get(URL+"/direct_assign", json=values)
    print(r.status_code)
    content = r.content.decode("utf-8").replace("\r\n","\n")
    buf = StringIO(content)
    
    content = pd.read_csv(buf, sep=";", index_col=0)
    
    print("Direct Assign:")
    print(content)
    
    return content

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
    differentiate_test(index_dict, truth_dict, eq_dict)
    compared = compare_test(truth_dict, eq_dict)
    assigned = assign_test(compared)
    direct_assign_test(truth_dict, eq_dict)
    
if __name__ == '__main__':
    main()