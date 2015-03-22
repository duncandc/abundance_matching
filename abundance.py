

def abundance_function(x, weights, bins, xlog=True):
    """
    given objects with property 'x' and weights, return an abundance function
    
    Parameters
    ==========
    x: array_like
    
    weights: array_like
    
    bins: array_like
    
    Returns
    =======
    dndx: dn, x
    """
    
    if use_log==True:
        x = np.log10(x)
    
    if np.shape(weights)==():
        weights = np.array([weights]*len(x))
    
    n = np.histogram(x,bins,weights=weights)[0]
    bin_centers = (bins[:-1]+bins[1:])/2.0
    dx = bins[1:]-bins[:-1]
    dn = n/dx
    
    #remove bins with zero counts
    keep = (dn>0.0)
    dn = dn[keep]
    bin_centers = bin_centers[keep]
    
    if not _is_monotonic(dn,bin_centers):
        print("warning, function is not monotonic.")
        
    reverse = _is_reversed(dn,bin_centers)
    
    sorted_inds = np.argsort(dn)
    dn = dn[sorted_inds]
    if reverse==True:
        bin_centers = bin_centers[::-1]
    
    #if two values are exactly the same, add 0.5 an average count to one of them
    i=0
    avg_weight = np.mean(weights)
    print(dn)
    for val1,val2 in zip(dn[:-1],dn[1:]):
       if val1==val2:
           dn[i+1] = dn[i]+avg_weight/dx[i+1]*0.5
           dn[i] = dn[i]-avg_weight/dx[i+1]*0.5
       i+=1
    
    print(dn)
    
    #check
    if not _is_monotonic(dn,bin_centers):
        raise ValueError("abundance function could not be calculated.")
    
    return dn, bin_centers