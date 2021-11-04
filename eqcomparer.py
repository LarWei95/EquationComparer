'''
Created on 03.11.2021

@author: larsw
'''
import numpy as np
import pandas as pd
from copy import deepcopy

class EquationComparer ():    
    MEANABSDIFF_KEY = "MeanAbsDiff"
    MAXABSDIFF_KEY = "MaxAbsDiff"
    MEANMINMAXDIFF_KEY = "MeanMinMaxDiff"
    MSE_KEY = "MSE"
    RMSE_KEY = "RMSE"
    
    def __init__ (self, additional_metrics_dict={}, add_metrics_weight_dict={}):
        self._metrics_dict = {
                EquationComparer.MEANABSDIFF_KEY : EquationComparer._direct_compare,
                EquationComparer.MAXABSDIFF_KEY : EquationComparer._direct_max_compare,
                EquationComparer.MEANMINMAXDIFF_KEY : EquationComparer._minmax_compare, 
                EquationComparer.MSE_KEY : EquationComparer._mse_compare,
                EquationComparer.RMSE_KEY : EquationComparer._rmse_compare
            }
        self._metrics_weights_dict = {
                EquationComparer.MEANABSDIFF_KEY : 1.0,
                EquationComparer.MAXABSDIFF_KEY : 1.0,
                EquationComparer.MEANMINMAXDIFF_KEY : 1.0, 
                EquationComparer.MSE_KEY : 1.0,
                EquationComparer.RMSE_KEY : 1.0
            }
        
        self._metrics_dict.update(additional_metrics_dict)
        self._metrics_weights_dict.update(add_metrics_weight_dict)
    
    def compare (self, truths, equations):
            
        eq_comparations = {}
        
        for equation_key in equations:
            # One equation, multiple possible truths
            equation = equations[equation_key]
            comparation = self.compare_equation(truths, equation)
            eq_comparations[equation_key] = comparation
            
        # Equation -> "Score"[Metric, Truth]
        eq_comparations = EquationComparer.studentize_comparation_dict(eq_comparations)
        return eq_comparations
        
    def reduce_comparations (self, comp_dict):
        eq_comparations = EquationComparer.weight_reduce_comparation_dict (comp_dict, self._metrics_weights_dict)
        eq_comparations = pd.concat(eq_comparations.values(), axis=1)
        return eq_comparations
        
    def assign_equations_to_truths (self, reduced_comp_df):
        minindx = reduced_comp_df.idxmin(axis=0)
        mins = reduced_comp_df.min(axis=0)
        
        minindx.name = "Truth"
        mins.name = "Score"
        
        assignment = pd.concat([minindx, mins], axis=1)
        
        return assignment
    
    def compare_equation (self, truths, equation):
        truth_metrics = []
        
        for truth_key in truths:
            truth = truths[truth_key]
            
            metrics = EquationComparer._full_compare(truth, equation, self._metrics_dict)
            metrics.name = truth_key
            truth_metrics.append(metrics)
            
        truth_metrics = pd.concat(truth_metrics, axis=1).transpose()
        return truth_metrics
        
    @classmethod
    def _full_compare (cls, truth, equation, metrics_dict):
        truth, equation = cls._fit_lengths(truth, equation)
        
        compared_metrics = {}
        
        for metrics_key in metrics_dict:
            # Scalar
            metric = metrics_dict[metrics_key](truth, equation)
            compared_metrics[metrics_key] = metric
            
        return pd.Series(compared_metrics)
        
    @classmethod
    def _direct_compare (cls, truth, mock):
        diff = np.abs(truth - mock)
        diff = np.mean(diff)
        return diff
    
    @classmethod
    def _direct_max_compare (cls, truth, mock):
        diff = np.abs(truth - mock)
        diff = np.max(diff)
        return diff
    
    @classmethod
    def _minmax_compare (cls, truth, mock):
        truthmin = np.nanmin(truth)
        truthmax = np.nanmax(truth)
        
        mockmin = np.nanmin(mock)
        mockmax = np.nanmax(mock)
        
        truth = (truth - truthmin) / (truthmax - truthmin)
        mock = (mock - mockmin) / (mockmax - mockmin)
        return cls._direct_compare(truth, mock)
    
    @classmethod
    def _mse_compare (cls, truth, mock):
        truth = np.mean((truth - mock) ** 2)
        return truth
    
    @classmethod
    def _rmse_compare (cls, truth, mock):
        return np.sqrt(cls._mse_compare(truth, mock))
        
    @classmethod
    def studentize_comparation_dict (cls, comp_dict):
        full = pd.concat(comp_dict.values(), axis=0)
        
        means = full.mean(axis=0)
        stds = full.std(axis=0)
        
        comp_dict = {
                k : (comp_dict[k] - means) / stds
                for k in comp_dict
            }
        return comp_dict
    
    @classmethod
    def weight_reduce_comparation_dict (cls, comp_dict, weights_dict):
        weights_sum = np.sum(list(weights_dict.values()))
        new_comp_dict = {}
        
        for equation_key in comp_dict:
            equation_comp = comp_dict[equation_key]
            
            for col_key in weights_dict:
                equation_comp[col_key] *= weights_dict[col_key]
                
            equation_comp = equation_comp.sum(axis=1) / weights_sum
            equation_comp.name = equation_key
            
            new_comp_dict[equation_key] = equation_comp
            
        return new_comp_dict
    
    @classmethod
    def _fit_lengths (cls, a, b):
        alen = len(a)
        blen = len(b)
        
        if alen < blen:
            b = b[:alen]
        elif alen > blen:
            a = a[:blen]
            
        return a, b
    
    @classmethod
    def calculate_differentials (cls, indices, values_dict, count):
        additionals = {}
        
        for index_key in indices:
            indx = indices[index_key]
            
            for value_key in values_dict:
                values = values_dict[value_key]
                additionals.update(cls._differentiate_through(index_key, value_key, indx, values, count))
                
        return additionals
        
    @classmethod
    def _differentiate_through (cls, indx_name, values_name, indx, values, count):
        fm = values
        
        ext = {}
        
        for i in range(count):
            grad = cls._gradients_of_array(indx, fm)
            name = "{:s} {:d}D({:s})".format(values_name, i+1, indx_name)
            ext[name] = grad
            fm = grad
            
        fm = values
        
        for i in range(count):
            grad = cls._integrals_of_array(indx, fm)
            name = "{:s} {:d}I({:s})".format(values_name, i+1, indx_name)
            ext[name] = grad
            fm = grad
            
        return ext
    
    @classmethod
    def _gradients_of_array (cls, x, y):
        x, y = cls._fit_lengths(x, y)
        
        startx = x[:-1]
        starty = y[:-1]
        
        endx = x[1:]
        endy = y[1:]
        
        xdiff = endx - startx
        ydiff = endy - starty
        
        gradients = ydiff / xdiff
        return gradients
        
    @classmethod
    def _integrals_of_array (cls, x, y):
        x, y = cls._fit_lengths(x, y)
        
        startx = x[:-1]
        starty = y[:-1]
        
        endx = x[1:]
        endy = y[1:]
        
        xdiff = endx - startx
        ydiff = endy - starty
        
        bases = xdiff * starty
        tops = xdiff * ydiff / 2
        
        integrals = bases + tops
        return integrals
        