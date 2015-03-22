import numpy as np

def make_SHAM_mock(mock, P_xy, mock_prop='mvir', gal_prop='mstar', use_log_mock_prop=True):
    """
    make a SHAM mock given a halo catalogue.  
    
    Parameters
    ==========
    mock: array_like
        structured array containing halo catalogue
    
    P_xy: function
        probability function that returns probability of x_gal given y_halo
    
    mock_prop: string
        key into mock which returns the halo property to build the SHAM mock
    
    Returns
    =======
    mock: structured array
        mock with new column containing galaxy property gal_prop
        
    Notes
    =====
    The probability of galaxy property 'x' given a halo with property 'y', 
    where mock[mock_prop] returns halo property 'y'.
    """
    
    from numpy.lib.recfunctions import append_fields
    
    mock = mock.view(np.recarray)
    
    if use_log_mock_prop==True:
        y = np.log10(mock[mock_prop])
    else:
        y = mock[mock_prop]
    
    x = P_xy(y).rvs(len(mock))
    
    if gal_prop in mock.dtype.names:
        mock[gal_prop] = x
    else:
        mock = append_fields(mock,gal_prop,x)
    
    return mock



