'''
Created on 04.11.2021

@author: larsw
'''
from flask import Flask, request, jsonify
from eqcomparer import EquationComparer
import json
import numpy as np
from pprint import pprint
from io import StringIO
import pandas as pd

APP = Flask(__name__)
INDICES_KEY = "indices"
EQUATIONS_KEY = "equations"
TRUTHS_KEY = "truths"

VALUES_KEY = "values"
DIFFS_KEY = "diffs"

COMPARATIONS_KEY = "comparations"

EQCOMP = EquationComparer()


@APP.route("/compare")
def compare ():
    data = request.json
    
    try:
        truths = data[TRUTHS_KEY]
        eqs = data[EQUATIONS_KEY]
        
        truths = {
                k : np.array(truths[k])
                for k in truths
            }
        eqs = {
                k : np.array(eqs[k])
                for k in eqs
            }
    except Exception as e:
        return str(e)
    
    comparations = EQCOMP.compare(truths, eqs)
    comparations = EQCOMP.reduce_comparations(comparations)
    buf = StringIO()
    
    comparations.to_csv(buf, sep=";")
    
    return buf.getvalue()

@APP.route("/assign")
def assign ():
    data = request.json
    
    try:
        comparations = data[COMPARATIONS_KEY]
        comparations = StringIO(comparations)
        comparations = pd.read_csv(comparations, sep=";", index_col=0)
    except Exception as e:
        return str(e)
    
    comparations = EQCOMP.assign_equations_to_truths(comparations)
    buf = StringIO()
    
    comparations.to_csv(buf, sep=";")
    return buf.getvalue()

@APP.route("/direct_assign")
def direct_assign ():
    data = request.json
    
    try:
        truths = data[TRUTHS_KEY]
        eqs = data[EQUATIONS_KEY]
        
        truths = {
                k : np.array(truths[k])
                for k in truths
            }
        eqs = {
                k : np.array(eqs[k])
                for k in eqs
            }
    except Exception as e:
        return str(e)
    
    comparations = EQCOMP.compare(truths, eqs)
    comparations = EQCOMP.reduce_comparations(comparations)
    comparations = EQCOMP.assign_equations_to_truths(comparations)
    buf = StringIO()
    
    comparations.to_csv(buf, sep=";")
    return buf.getvalue()
    
@APP.route("/differentiate")
def differentiate ():
    diffcount = int(request.args.get(DIFFS_KEY, 0))
    data = request.json
    
    try:
        indices = data[INDICES_KEY]
        values = data[VALUES_KEY]
        
        indices = {
                k : np.array(indices[k])
                for k in indices
            }
        values = {
                k : np.array(values[k])
                for k in values
            }
    except Exception as e:
        return str(e)
        
    additionals = EquationComparer.calculate_differentials(indices, values, diffcount)
    additionals = {
            k : additionals[k].tolist()
            for k in additionals
        }
    return jsonify(additionals)

if __name__ == "__main__":
    APP.run()