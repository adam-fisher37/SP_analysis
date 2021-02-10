import numpy as np
import os
import re
import fnmatch as fm
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('xtick', direction='in', top=True)
mpl.rc('ytick', direction='in', right=True)
mpl.rc('xtick.minor', visible=True)
mpl.rc('ytick.minor', visible=True)

def scan_dict(dick,path,wc='*.txt',skips=21,cols=(0,1),bad_bois=False):
    '''
    imports all scans from a specific directory folder and puts them in the inputted dictionary. each key gives the sample #, scan #, and x/z
    also can handle flipping the data so that the glass side is on the left, which is nesecary for the rest of the functions
    uses np.loadtxt as of 9/5/20, currently also made for scans so the default args are for that, this can be co-opted for other shit but will be a lil annoyting potentially
    inputs:
    dictionary - dict - dict you wish to fill with scan data, should be empty
    path - str - relative path to scan data
    wc - string, optional - the wildcard that you want to use, any files that match the wildcard will be tried to put into the dictionary, NOTE: anything in your wildcard is stripped from the file name when it is saved as the key for the dictionary so tbh probs dont wanna mess with it. default is '*.txt'
    bad_bois - array of strings, optional - an array with each element being a string for a key of the data that youd like to flip. Default is to not flip anything. ex: np.array(['s3t4', 's3t5', 's3t6', 's6t1', 's6t2', 's6t3'])
    outputs:
    dictionary - dict - the dict you put in but with stuff now
    '''
    #load the data and put into the dict
    # have mutiple sets that need to be flipped
    if (hasattr(bad_bois,'T')):
        assert(bad_bois.dtype=='<U4')
        for fname in os.listdir(path):
            if fm.fnmatch(fname,wc):
                dick[fname.strip(wc.strip('*'))] = np.loadtxt(path+fname,skiprows=skips,usecols=cols,unpack=True)
    # flip the bois
                if fname.strip(wc.strip('*')) in bad_bois:
                    dick[fname.strip(wc.strip('*'))][1] = dick[fname.strip(wc.strip('*'))][1][::-1]
    elif (isinstance(bad_bois,str)):
        for fname in os.listdir(path):
            if fm.fnmatch(fname,wc):
                dick[fname.strip(wc.strip('*'))] = np.loadtxt(path+fname,skiprows=skips,usecols=cols,unpack=True)
                if (fname.strip(wc.strip('*'))==bad_bois):
                    dick[fname.strip(wc.strip('*'))][1] = dick[fname.strip(wc.strip('*'))][1][::-1]
    elif (isinstance(bad_bois,bool)):
        for fname in os.listdir(path):
            if fm.fnmatch(fname,wc):
                dick[fname.strip(wc.strip('*'))] = np.loadtxt(path+fname,skiprows=skips,usecols=cols,unpack=True)
    else:
        print('I dont think you have the facilities for that big man')
    return dick

def make_x_micro(dick):
    '''
    helper function to make all data sets for x-direction have units of micrometers
    run before doing any adv functions that add to the dictionary
    ASSUMES that if the final value of x-direction is >2, then it is initally nanometers
    '''
    for key in dick:
        if (dick[key][0][-1]>2.):
            dick[key][0] = dick[key][0]/1000.
    return dick

def get_ind(dick,key,pos):
    'get the index of the input position for a specific key, as they are not all equally spaced, also accounts for the array not starting at 0, NOTE: pos [micrometers]'
    dx = dick[key][0][1] - dick[key][0][0]
    dist = pos - dick[key][0][0]
    ind = int(dist//dx)
    return ind

def key_plot(dick,key,end_dist=None):
    '''
    plot a x vs z graph from one of the scans from certian sample's dictionary
    uses the key names to plot
    only plots, NO OUTPUTS
    inputs:
    dick - dict - dictionary from which you would like to plot
    key - str - key name of the run you want to plot
    end_dist - float OR tuple of floats - the end position in mircons or range over which you wish to see the sample plot, if tuple should be in form (begining of range,end of range)
    returns nice graph
    '''
    if key in dick:
        if (end_dist==None):
            plt.figure();
            plt.plot(dick[key][0],dick[key][1],'k.',ms=1);
            plt.xlabel('Horz. Dist. ($\mu$m)');
            plt.ylabel('Vert. Dist. (nm)');
            plt.title('x vs. z for:'+key);
        elif (isinstance(end_dist,float)):
            ind = get_ind(dick,key,end_dist)
            plt.figure();
            plt.plot(dick[key][0][:ind],dick[key][1][:ind],'k.',ms=1);
            plt.xlabel('Horz. Dist. ($\mu$m)');
            plt.ylabel('Vert. Dist. (nm)');
            plt.title('x vs. z for:'+key);
        elif (isinstance(end_dist,tuple)):
            ind_a = get_ind(dick,key,end_dist[0])
            ind_b = get_ind(dick,key,end_dist[1])
            plt.figure();
            plt.plot(dick[key][0][ind_a:ind_b],dick[key][1][ind_a:ind_b],'k.',ms=1);
            plt.xlabel('Horz. Dist. ($\mu$m)');
            plt.ylabel('Vert. Dist. (nm)');
            plt.title('x vs. z for:'+key);
        else:
            print('no dice champ')
    else:
        print('no dice champ')
    return

def linf(x,m,b):
    '''
    good ole linear function for fitting the scan data
    inputs:
    x - 1-d array - the x data of a scan
    m - float - slope
    b - float - offset
    '''
    return m*x + b
def proj(dick,key,xf,xi=None):
    '''
    projects the scan onto the x-axis
    either fit the glass part and gather the residuals, which should be on the x-axis
    inputs:
    dick - dict - dict from which youre working in
    key - str - run you want to make not look dumb af
    xf - float - fitting line from x[xi_ind] to this x position [microns], WANT THIS TO BE ON THE GLASS OR FLAT PART TOO
    xi - float - where to start the line fit, default = .03 microns from 'scan_dict', shoudnt need cuz you can always just make xf shorter but hey its here for YOU man
    outputs: 
    returns the dictionary that you input with the added key that is a tuple of x coord and the adjusted z coords under the key name of: key+'_adj'
    '''
    # load data cuz i want to iter over a whole ass dict, but dont wanna call it like tht every time
    (x,z) = dick[key]
    # find value you want to make your second coord pair to calc slope from
    xf_ind = get_ind(dick,key,xf)
    if (xi==None):
        m = (z[0]-z[xf_ind])/(x[0]-x[xf_ind])
        # curve fitting time bby
        p0 = np.array([m,z[0]])
        sig_fact = np.mean(np.diff(z[0:xf_ind])) # nm, use roughness from measurment, convert from angstroms, but usually less than 2 nm, so we can use absolute sigma in fit
        sig = np.ones(xf_ind)*sig_fact
        (pg,Cg) = opt.curve_fit(linf,x[:xf_ind],z[:xf_ind],p0=p0,sigma=sig,absolute_sigma=True)
        resid = z - linf(x,*pg)
    else:
        xi_ind = get_ind(dick,key,xi)
        m = (z[xi_ind]-z[xf_ind])/(x[xi_ind]-x[xf_ind])
        # curve fitting time bby
        p0 = np.array([m,z[xi_ind]])
        sig_fact = np.mean(np.diff(z[xi_ind:xf_ind])) # nm, use roughness from measurment, convert from angstroms, but usually less than 2 nm, so we can use absolute sigma in fit
        sig = np.ones_like(z[xi_ind:xf_ind])*sig_fact
        (pg,Cg) = opt.curve_fit(linf,x[xi_ind:xf_ind],z[xi_ind:xf_ind],p0=p0,sigma=sig,absolute_sigma=True)
        resid = z - linf(x,*pg)
    dick[key+'_adj'] = (x,resid)
    return dick

def height(dick,key,xf_a,xi_b,xf_b,xi_a=None,result=False):
    '''
    returns the difference in height btwn the two materials for a specific run along with error. does this by taking data from xi_a to xi_b which is left of the jump and averages that height and takes that difference from the averaged height from (xf_a,xf_b). you will need to look at this data beforehand to know where the positions are that you want to use. 
    (i guess theoretically you wouldnt need to )
    inputs:
    dick - dictionary - dict from which you are working from
    key - str - data set you want to get the height from, NOTE you should have run proj for this set but just call the key 's#t#', hopefully should be able to handle it as long as it has that part in there, perfer if its: 's#t#_adj'
    xi_a - float, default None - averages the glass height from here to xf_a, defaults to the start of the horziontal coords
    xf_a - float - end of range of coords to avg the glass height for [microns]
    xi_b - float - average the height of the sample (al2o3, etc.) from this coord to xf_b, start after the jump
    xf_b - float - end of the range of sample
    result - bool - default False, makes final plot of key_adj with horz line at the height, and prints the height with its std 
    outputs:
    prints the height along with the uncertianty [nm]
    also shows a plot of the whole run, with a horz line where the height is
    and then adds the height and std to a new key named: 's#t#_h'
    '''
    if (xi_a==None):
        a_ind = (0,get_ind(dick,key,xf_a))
    elif (isinstance(xi_a,float)):
        assert(xi_a<xf_a)
        a_ind = (get_ind(dick,key,xi_a),get_ind(dick,key,xf_a))
    else:
        raise Exception('xi_a has an incorrect input')
    assert((xf_a<xi_b))
    assert(xi_b<xf_b)
    b_ind = (get_ind(dick,key,xi_b),get_ind(dick,key,xf_b))
    if (key == key.strip('_adj')):
        key = key + '_adj'
    gl_avg = np.mean(dick[key][0][a_ind[0]:a_ind[1]])
    al_avg = np.mean(dick[key][1][b_ind[0]:b_ind[1]])
    gl_std = np.std(dick[key][0][a_ind[0]:a_ind[1]],ddof=2)
    al_std = np.std(dick[key][1][b_ind[0]:b_ind[1]],ddof=2)
    h = al_avg - gl_avg
    pm = np.sqrt((gl_std**2)+(al_std**2))
    if result:
        key_plot(dick,key)
        plt.axhline(h);
        print('height of ',key,' is:',h,'+-',pm,'nm')
    dick[key.strip('adj')+'h'] = (h,pm)
    return dick

def sp_analysis_hard(dick,n_samp,xf_a,xi_b,xf_b,xi_a=False):
    '''
    function that uses proj and height on all data sets and will average for each sample
    wont need to do all by hand but will need to look at each og data graph for values
    NOTE: for best results look at the adj data graph for input values
    made currently (9/5/20) to start with only OG data sets, no _adj's or _h's
    also cannot for any input just input a float, must be an array even if all the same number
    NOTE: the program as of (9/9/20) will OVERWRITE the keys '_adj', '_h', 'avg' EVERY time the program is run, it will just overwrite with the same results if you dont change any inputs but you should keep that in mind
    NOTE: if you change the name of the result key, currently 's#avg', then it will still keep the prev name key, restart kernel and clear outputs to fix
    inputs:
    dick - dict - dict you are working in
    n_samp - float/int - number of samples, aka if s8 is highest s#, then n_samp=8
    xi_a - array of floats - default False, 1xN (N = # of data sets) array the start of the fit/avg on the glass side
    xf_a - array of floats - 1xN array or float [microns, both fits line in proj and avg in height on the glass side
    xi_b - array of floats - 1xN array [microns], starts the avg in height on the material side, start after the jump and decay thing
    xf_b - array of floats - 1xN array [microns], take the avg from xi_b to xf_b on material side
    outputs:
    returns the dictionary with the adjusted z's for each set and the heights in their corresponding key [nm]
    as well as the an average height for each sample, under key: 's#avg'
    '''
    # make an list/array to iterate over
    # imports all the names of the raw data keys into a array of strings
    bois = np.array([key for key in dick if (re.search('adj|h|_|avg',key)==None)])
    #assert that the lengths of the arrays matches so it doesnt throw any loop errors
    assert(len(bois)==len(xf_a)==len(xi_b))
    # NOTE: need to have imported numpy as 'np'
    # will have the same basic loop to do all the ops for each instance of the input for xi_a
    for i in range(len(bois)):
        key = bois[i]
        if (type(xi_a)==bool):
            dick = proj(dick,key,xf_a[i])
            dick = height(dick,key,xf_a[i],xi_b[i],xf_b[i])
        elif (type(xi_a)==float):
            assert(xi_a<xf_a[i])
            dick = proj(dick,key,xf_a[i],xi=xi_a)
            dick = height(dick,key,xf_a[i],xi_b[i],xf_b[i],xi_a=xi_a)
        elif (type(xi_a)==np.ndarray):
            assert(len(bois)==len(xi_a))
            assert(xi_a[i]<xf_a[i])
            dick = proj(dick,key,xf_a[i],xi=xi_a[i])
            dick = height(dick,key,xf_a[i],xi_b[i],xf_b[i],xi_a=xi_a[i])
        else:
            raise Exception('xi_a has incorrect input')
    # now need to average across the samples and put them into new key
    avg_arr = np.arange(1,n_samp+1)
    std_arr = np.zeros_like(avg_arr)
    # starts as the sample number i want, thn element gets overwritten as the avg
    for i in range(len(avg_arr)):
        samp_str = 's'+str(avg_arr[i])+'.._h'
        s = 0.
        std = 0.
        cts = 0
        for key in dick:
            if re.search(samp_str,key):
                s += dick[key][0]
                std += (dick[key][1])**2
                cts += 1
        avg_arr[i] = s/cts
        std_arr[i] = np.sqrt(std)
        dick['s'+samp_str[1]+'avg'] = (avg_arr[i],std_arr[i])
    return dick