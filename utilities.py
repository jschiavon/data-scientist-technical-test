import numpy as np
import pandas as pd


def clean_money_columns(dataset, money_columns, log: bool = False, scale: bool = False, indicator: bool = True, verbose: bool = False):
    """
    Clean up and transform a column with monetary values in a floating point valued column.
    
    Parameters
    ----------
    dataset : pandas DataFrame
        The dataframe to be converted
    money_columns : list of str
        The list of columns names that should undergo conversion
    log : {`True`, `False`}, optional
        A boolean flag that controls if the log of the columns should be returned. Default `False`
    scale : {`True`, `False`}, optional
        A boolean flag that controls whether to scale (via standard scaling) or not the values on the column around the average. Default `False`
    indicator : {`True`, `False`}, optional
        A boolean flag that controls if an indicator column should be added for each feature to signify that the value was exactly 0. Default `True`
    verbose : {`True`, `False`}, optional
        A boolean flag that controls verbosity of the method. Default `False`
    
    Returns
    -------
    pandas DataFrame
        The converted DataFrame
    """
    eps = 1.
    for name, column in dataset.items():
        if name in money_columns:
            column = column.str.replace("$", "").str.replace(",","")
            column = column.astype(float)
            if indicator:
                dataset[name+'_ind'] = (column == 0).astype('boolean')
                dataset.loc[pd.isna(column), name+'_ind'] = pd.NA
            if log:
                column = np.log(column + eps)
            if scale:
                column = (column - column.mean()) / column.std()
            dataset[name] = column
        if verbose: print(f"{name} is a {column.dtype} column")
        
    return dataset


def clean_types(dataset, verbose: bool = False):
    """
    Clean up and standardize the types of the columns.
    
    Parameters
    ----------
    dataset : pandas DataFrame
        The dataframe to be converted
    verbose : {`True`, `False`}, optional
        A boolean flag that controls verbosity of the method. Default `False`
    
    Returns
    -------
    pandas DataFrame
        The converted DataFrame
    """
    categoricals = {}
    numericals = {}
    booleans = {}
    for name, column in dataset.items():
        if column.dtype in [object, 'string']:
            categoricals[name] = column.astype('category')
            if verbose: print(f"{name:<10} is a {str(categoricals[name].dtype):<8} column")
        elif column.dtype in [bool, 'boolean']:
            booleans[name] = column
            if verbose: print(f"{name:<10} is a {str(booleans[name].dtype):<8} column")
        else:
            numericals[name] = column
            if verbose: print(f"{name:<10} is a {str(numericals[name].dtype):<8} column")
    dataset = pd.concat([
        pd.DataFrame(numericals, index=dataset.index),
        pd.DataFrame(categoricals, index=dataset.index),
        pd.DataFrame(booleans, index=dataset.index)
    ], axis=1)
    return dataset