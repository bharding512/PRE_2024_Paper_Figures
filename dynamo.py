# A class to solve the dynamo equation, using the electrostatic approximation.
# This is almost entirely informed by Richmond [1995].
# This code has been validated against WACCM-forced-TIEGCM. Using the same 3D winds and conductivities, I get the same potential.

# Inputs:
#   - wind
#   - conductivity
#   - mlat, mlon grid
# Outputs:
#   - Potential, from which the electric fields, plasma drifts, currents, etc., can be calculated.

# Internally:
# - distances are in m
# - coordinates that end in "e" refer to the edge grid, while the rest refer to the midpoint grid
# - Potential (Phi) is defined on edges, and currents are generally defined at midpoints (though be careful)
# - x_midpoint[m] > x_edge[m] for both lat and lon
# - modified apex coordinates are used (Richmond, 1995), big thanks to apexpy.

__version__ = '2.0.1'

# Version 2 is updating to use xarray Dataset, because:
# (1) It will make keeping track of dimensions easier
# (2) It will make extending to multiple times easier
# Instead of having Dynamo be a class, the fundamental unit will be a Dataset in standard form, and this 
# module will contain functions that operate on that Dataset. This dataset will be "d" throughout the code.
# Version 2.0.1 is a trimmed down version that was pushed to Github/Zenodo for the 2024 PRE paper.


from datetime import datetime, timedelta
import pyglow # It would be nice to remove pyglow dependence at some point. It's just used for |B| which might be derivable from apexpy.
import os
import glob
import numpy as np
import gc
import apexpy
import xarray as xr
import copy
import pandas as pd
from scipy import interpolate

import scipy.sparse as sp
import scipy.sparse.linalg # Why oh why is this necessary

import sys
import time as time_module
from IPython.display import display, clear_output
def printerase(s):
    clear_output(wait=True)
    time_module.sleep(0.01)
    print(s)
    sys.stdout.flush()
    


def compute_slt(dt, longit):
    '''
    Compute solar local time. From  https://stackoverflow.com/questions/13314626/local-solar-time-function-from-utc-and-longitude/13425515

    dt = datetime.datetime
    longit = deg, longitude
    
    Returns:
    solar_time in hrs
    '''
    from math import pi, cos, sin
    from datetime import time

    gamma = 2 * pi / 365 * (dt.timetuple().tm_yday - 1 + float(dt.hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * cos(gamma) - 0.032077 * sin(gamma) \
             - 0.014615 * cos(2 * gamma) - 0.040849 * sin(2 * gamma))
#     print eqtime
    decl = 0.006918 - 0.399912 * cos(gamma) + 0.070257 * sin(gamma) \
           - 0.006758 * cos(2 * gamma) + 0.000907 * sin(2 * gamma) \
           - 0.002697 * cos(3 * gamma) + 0.00148 * sin(3 * gamma)
    time_offset = eqtime + 4 * longit
    tst = dt.hour * 60 + dt.minute + dt.second / 60 + time_offset
    solar_time = datetime.combine(dt.date(), time(0)) + timedelta(minutes=tst)
    slt_hrs = (solar_time - datetime(solar_time.year, solar_time.month, solar_time.day)).total_seconds()/3600.
    return slt_hrs



def wgs84constants():
    '''
    Return the constants associated with the WGS-84 coordinate system
    
    OUTPUTS:
    
      * a - semi-major axis of Earth, km
      * b - semi-minor axis of Earth, km
      * e - eccentricity of Earth
      
    '''
    
    # http://en.wikipedia.org/wiki/World_Geodetic_System (Nov 10, 2014)
    # https://en.wikipedia.org/wiki/Flattening (Sep 9, 2015) 
    a = 6378.137 # semi-major axis of earth [km]
    inv_f = 298.257223563
    f = 1.0/inv_f
    b = a*(1-f) # semi-minor axis of earth [km]
    e = np.sqrt(1-b**2/a**2) # eccentricity of earth
    return a,b,e



def ecef_to_wgs84(ecef_xyz):
    '''
    Convert from earth-centered earth-fixed (ECEF)
    coordinates (x,y,z) to WGS-84 latitude, longitude, and altitude.
    INPUTS:
    
     *  ecef_xyz - a length-3 array containing the X, Y, and Z locations in ECEF
                   coordinates in kilometers.

    OUTPUTS:
     *  latlonalt - a length-3 array containing the WGS-84 coordinates:
                    [latitude (degrees), longitude (degrees), altitude (km)]
                    Altitude is defined as the height above the reference
                    ellipsoid. Longitude is defined in [0,360).

    HISTORY:
     * 11-Jun-2006: Initial MATLAB template created by Jonathan J. Makela
       (jmakela@uiuc.edu)
     * 17-July-2006: Algorithm implemented by Dwayne P. Hagerman
       (dhagerm2@uiuc.ed)
     * 10-Nov-2014: Translated to Python by Brian J. Harding
       (bhardin2@illinois.edu)
     * 19-Jan-2015: Changed from iterative to closed-form implementation (BJH)
     * 09-Sep-2015: Changed from ublox implementation (with mm accuracy) to
                    Olson implementation (with nm accuracy). (BJH)
                    https://possiblywrong.wordpress.com/2014/02/14/when-approximate-is-better-than-exact/
    '''
    
    
    # WGS-84 ellipsoid parameters
    a,b,_ = wgs84constants()
    f = 1-b/a

    # Derived parameters
    e2 = f * (2 - f)
    a1 = a * e2
    a2 = a1 * a1
    a3 = a1 * e2 / 2
    a4 = 2.5 * a2
    a5 = a1 + a3
    a6 = 1 - e2

    def ecef_to_lla(ecef):
        """Convert ECEF (meters) to LLA (radians and meters).
        """
        # Olson, D. K., Converting Earth-Centered, Earth-Fixed Coordinates to
        # Geodetic Coordinates, IEEE Transactions on Aerospace and Electronic
        # Systems, 32 (1996) 473-476.
        w = math.sqrt(ecef[0] * ecef[0] + ecef[1] * ecef[1])
        z = ecef[2]
        zp = abs(z)
        w2 = w * w
        r2 = z * z + w2
        r  = math.sqrt(r2)
        s2 = z * z / r2
        c2 = w2 / r2
        u = a2 / r
        v = a3 - a4 / r
        if c2 > 0.3:
            s = (zp / r) * (1 + c2 * (a1 + u + s2 * v) / r)
            lat = math.asin(s)
            ss = s * s
            c = math.sqrt(1 - ss)
        else:
            c = (w / r) * (1 - s2 * (a5 - u - c2 * v) / r)
            lat = math.acos(c)
            ss = 1 - c * c
            s = math.sqrt(ss)
        g = 1 - e2 * ss
        rg = a / math.sqrt(g)
        rf = a6 * rg
        u = w - rg * c
        v = zp - rf * s
        f = c * u + s * v
        m = c * v - s * u
        p = m / (rf / g + f)
        lat = lat + p
        if z < 0:
            lat = -lat

        lat = 180./np.pi * lat
        lon = 180./np.pi * math.atan2(ecef[1], ecef[0])
        if lon < 0.0:
            lon += 360.
        alt = f + m * p / 2
        return np.array([lat, lon, alt])
    
    return ecef_to_lla(ecef_xyz)

def wgs84_to_ecef(latlonalt):
    '''
    Convert from WGS84 coordinates [latitude, longitude, altitude] to 
    earth-centered earth-fixed coordinates (ECEF) [x,y,z]
    
    INPUTS:
      * latlonalt - a length-3 array containing the WGS-84 coordinates:
                    [latitude (degrees), longitude (degrees), altitude (km)]    

    OUTPUTS:
      * ecef_xyz - a length-3 array containing the X, Y, and Z locations in ECEF
                   coordinates in kilometers.
                  
    HISTORY:
      * 11-Jun-2006: Initial MATLAB template created by Jonathan J. Makela
        (jmakela@uiuc.edu)
      * 17-July-2006: Algorithm implemented by Dwayne P. Hagerman
        (dhagerm2@uiuc.ed)
      * 10-Nov-2014: Translated to Python by Brian J. Harding
        (bhardin2@illinois.edu)    
    '''
    
    a,b,e = wgs84constants()
    
    lat = latlonalt[0]*np.pi/180.
    lon = latlonalt[1]*np.pi/180.
    alt = latlonalt[2]*1.0
    
    x = a * np.cos(lon) / np.sqrt(1 + (1-e**2) * np.tan(lat)**2) + alt*np.cos(lon)*np.cos(lat)
    y = a * np.sin(lon) / np.sqrt(1 + (1-e**2) * np.tan(lat)**2) + alt*np.sin(lon)*np.cos(lat)
    z = a * (1-e**2) * np.sin(lat) / np.sqrt(1 - e**2 * np.sin(lat)**2) + alt*np.sin(lat)
    
    return np.array([x,y,z])
  



def qlat_to_mlat(qlat, h, hR, RE):
    '''
    Convert quasi-dipole latitude to modified apex latitude
    
    INPUTS:
    qlat = [deg], array-like, quasi-dipole latitude
    h   = [m],   array-like, altitude 
    hR  = [m],   scalar. reference altitude for modified apex coordinates
    RE  = [m],   scalar. Earth radius
    
    OUTPUTS:
    mlat = [deg], array-like, modified apex latitude
    '''
    if h < 10e3:
        print('WARNING: Are you sure you\'re specifying h in meters?')
    qlat[qlat==0] = 1e-6 # This is a shortcut to using the actual formula

    cosq = np.cos(qlat*np.pi/180.)
    cosm = cosq * ((RE+h)/(RE+hR))**(-0.5)
    absmlat = 180./np.pi * np.arccos(cosm)
    mlat = np.sign(qlat)*absmlat
    return mlat




def init(h, hR=80e3, tref = datetime(2020,3,24,14,0,0), mlonde_res = 15., mlatde_res=3., mlatde_max = 90.):
    '''
    Initializes the dynamo xarray.Dataset "d". Setup the grid such that the grid midpoints are field lines that have the given apex heights.
    This uses an extrapolation to higher heights/mlats.
    These altitudes are where the wind and conductivity will be defined.

    INPUTS:
        h: [m], apex heights, length P

    OPTIONAL INPUTS:
        hR:       [m], reference altitude for modified apex coordinates
        tref:     [datetime], used for magnetic field model, and possibly conductivity and wind
        mlonde_res:    [deg], spacing of the mlon grid. Choose this to divide 360 evenly.
        mlatde_res:    [deg], nominal spacing of the mlat grid above the top specified apex height. The code does not respect this
                              exactly, since it enforces a smooth transition and a maximum specified mlat edge (see below).
        mlatde_max:    [deg], The upper *edge* boundary of the mlatde grid. (Default = max = 90)
        
    OUTPUT:
        d: standard dynamo Dataset

    TO-DO:
     - this is a long function. If there's ever a need to split out the component parts, I should do that.

    VARIABLES
    # mlat is lam variable, or l
    # mlon is phi variable or p
    # alt  is geodetic altitude variable or h

    ## 2D variables defined on midpoint grid:
    # KpD - zonal wind-driven current term
    # KlD - meridional/vertical wind-driven current term
    # Spp, Sll, Spl, Slp - conductance terms

    ## 2D variables defined on edge grid:
    # Phi - potential (to be solved for)

    ## 3D variables
    # These are needed to compute K, S
    # Separate arrays are needed for the N and S hemispheres
    # These are all defined on midpoint grids
    # d1, d2, d3, e1, e2, e3 -- shape (M,N,P,3) -- modified apex coordinate base vectors
    # sigP, sigH -- shape (M,N,P) -- conductivities 
    # ug -- shape (M,N,P,2) (wind in geodetic coordinates, ENU, but vertical wind is assumed 0 and omitted)
    '''
    
    h = np.array(h) # This is necessary because passing a DataArray leads to odd behavior

    d = xr.Dataset()

    yr = tref.year + (tref - datetime(tref.year, 1, 1)).days/365. # year with fraction 
    d.attrs['apex_obj'] = apexpy.Apex(date=yr, refh=hR/1e3)
    d.attrs['RE'] = 1e3 * d.apex_obj.RE # [m], mean Earth radius (Richmond, 1995)
    d.attrs['hR'] = hR
    d.attrs['tref'] = tref

    assert mlonde_res > 0.
    assert mlatde_res > 0.
    assert mlatde_max <= 90.
    assert h[0] > d.hR, "Reference altitude must be below the grid"

    tstart = datetime.now()

    ######## Build 1D grids #########
    mlonde = np.arange(-180, 180+mlonde_res, mlonde_res)[:-1]
    mlond  = mlonde + mlonde_res/2.
    mlond[mlond > 180] -= 360.

    # Build mlat grid. This does *not* set the midpoints to be the average of the edges. (At first I thought this
    # was a requirement, but after some testing I think I was wrong. As long as the variables are well resolved in
    # regions of large gradients, it's ok to have an irregular grid.)
    mlatd_sub0 = apexht_to_mlatd(d, h) # The mlats specified
    P = len(h)
    # Create the edge points for this grid
    mlatde_sub0 = np.concatenate(([mlatd_sub0[0] - (mlatd_sub0[1] - mlatd_sub0[0])/2.], # A half-step before
                                 (mlatd_sub0[1:] + mlatd_sub0[:-1])/2,  # Midpoints
                                 [mlatd_sub0[-1] + (mlatd_sub0[-1] - mlatd_sub0[-2])/2.] # A half-step after
                               ))
    # Extrapolate this edge grid to the high latitudes
    dlat0 = mlatd_sub0[-1] - mlatd_sub0[-2]
    print(mlatde_sub0)
    # Use a 6-grid transition to the requested resolution
    mlatde_sub1 = mlatde_sub0[-1] + np.cumsum(np.linspace(dlat0, mlatde_res, 7))
    MM = int(np.ceil((mlatde_max - mlatde_sub1[-2])/mlatde_res)) # number of remaining grids
    if MM<0:
        raise Exception('mlatde_res too coarse. Re-run or make this code smarter')
    mlatde_sub2 = np.linspace(mlatde_sub1[-1], mlatde_max, MM)
    # Put them together
    mlatde = np.concatenate((mlatde_sub0, mlatde_sub1[:-1], mlatde_sub2))
    mlatd  = np.concatenate((mlatd_sub0, [mlatd_sub0[-1]+dlat0], (mlatde[P+2:] + mlatde[P+1:-1])/2.))

    # Decision: The radians variables will be the coordinates (more natural) and the degrees will be variables. 
    # I believe this will allow for an easy way to swap_dims if necessary
    d['mlate']  = mlatde*np.pi/180. # [rad] Should these be coordinates or variables? 
    d['mlone']  = mlonde*np.pi/180. # [rad]
    d['mlat']   = mlatd*np.pi/180. # [rad]
    d['mlon']   = mlond*np.pi/180. # [rad]

    d['mlatde'] = (['mlate'], mlatde)
    d['mlatd']  = (['mlat'], mlatd)
    d['mlond']  = (['mlon'], mlond)
    d['mlonde'] = (['mlone'], mlonde)

    hAe = mlatd_to_apexht(d, d.mlatde.values)
    assert hAe[0] > d.hR, "Bottom of edge grid is below the apexpy reference altitude"
    hA  = mlatd_to_apexht(d, mlatd)
    alt = hA[:P] #  This is important to enforce exactly, to avoid numerical rounding issues at the apex of a field line.

    d['alt'] = alt # This is the coordinate
    d['hAe'] = (['mlate'], hAe) # [m] apex altitude for each mlat edge
    d['hA']  = (['mlat'], hA) # [m] apex altitude for each mlat midpoint

    d['mlt'] = (['mlon'], d.apex_obj.mlon2mlt(d.mlond, d.tref))
    _, lon_at_mageq_at_hR = d.apex_obj.convert(0, d.mlond, 'apex', 'geo', height=d.hR/1e3, )
    d['slteq'] = (['mlon'], np.array([compute_slt(d.tref, lon) for lon in lon_at_mageq_at_hR]))

    M = len(mlatd)
    N = len(mlond)
    d.attrs['M'] = M
    d.attrs['N'] = N
    d.attrs['P'] = P

    

    ######### Build geodetic location arrays glat, glon arrays -- these need to be 3D #########
    # These are defined at the midpoints. Separate arrays are held for the N and S hemispheres.
    GLATDN = np.nan * np.zeros((M,N,P)) # [deg], geodetic location of each point in grid
    GLONDN = np.nan * np.zeros((M,N,P))
    GLATDS = np.nan * np.zeros((M,N,P)) # [deg], geodetic location of each point in grid
    GLONDS = np.nan * np.zeros((M,N,P))
    MLATD, MLOND = np.meshgrid(mlatd, mlond, indexing='ij')
    for p in range(P):
        printerase('3D grid: %i / %i' % (p+1, P))
        # Take care to handle the point at the apex.
        idx_below = hA >= alt[p] # Care was taken above to make sure numerical rounding doesn't mess this up at the apex height
        # Perform computation for all points at or below apex
        glatp, glonp = d.apex_obj.convert( MLATD[idx_below,:], MLOND[idx_below,:], 'apex', 'geo', height=alt[p]/1e3)
        GLATDN[idx_below,:,p] = glatp
        GLONDN[idx_below,:,p] = glonp
        glatp, glonp = d.apex_obj.convert(-MLATD[idx_below,:], MLOND[idx_below,:], 'apex', 'geo', height=alt[p]/1e3)
        GLATDS[idx_below,:,p] = glatp
        GLONDS[idx_below,:,p] = glonp

    d['GLATDN'] = (['mlat','mlon','alt'], GLATDN)
    d['GLATDS'] = (['mlat','mlon','alt'], GLATDS)
    d['GLONDN'] = (['mlat','mlon','alt'], GLONDN)
    d['GLONDS'] = (['mlat','mlon','alt'], GLONDS)

    ######### Compute Apex coordinate base vectors ##########
    d1N  = np.nan * np.zeros((M,N,P,3)) # d1 vector expressed in ENU coordinates
    d1S  = np.nan * np.zeros((M,N,P,3))
    d2N  = np.nan * np.zeros((M,N,P,3))
    d2S  = np.nan * np.zeros((M,N,P,3))
    d3N  = np.nan * np.zeros((M,N,P,3))
    d3S  = np.nan * np.zeros((M,N,P,3))
    e1N  = np.nan * np.zeros((M,N,P,3)) # e1 vector expressed in ENU coordinates
    e1S  = np.nan * np.zeros((M,N,P,3))
    e2N  = np.nan * np.zeros((M,N,P,3))
    e2S  = np.nan * np.zeros((M,N,P,3))
    e3N  = np.nan * np.zeros((M,N,P,3))
    e3S  = np.nan * np.zeros((M,N,P,3))
    dsN  = np.nan * np.zeros((M,N,P)) # [m], Step size for Riemann integral
    dsS  = np.nan * np.zeros((M,N,P))
    ECEFN = np.nan * np.zeros((M,N,P,3)) # ECEF coordinates of each point
    ECEFS = np.nan * np.zeros((M,N,P,3))

    for p in range(P):
        printerase('Base vectors: %i / %i' % (p+1, P))
        # Take care to handle the point at the apex.
        idx_below = hA >= alt[p] # Care was taken above to make sure numerical rounding doesn't mess this up at the apex height
        # Perform computation for all points at or below apex
        _, _, _, _, _, _, d1p, d2p, d3p, e1p, e2p, e3p = d.apex_obj.basevectors_apex(GLATDN[idx_below,:,p].flatten(), 
                                                                                     GLONDN[idx_below,:,p].flatten(), alt[p]/1e3, coords='geo')
        shap = (sum(idx_below), N, 3) # The shape of the array to return
        d1N[idx_below,:,p,:] = d1p.T.reshape(shap)
        d2N[idx_below,:,p,:] = d2p.T.reshape(shap)
        d3N[idx_below,:,p,:] = d3p.T.reshape(shap)
        e1N[idx_below,:,p,:] = e1p.T.reshape(shap)
        e2N[idx_below,:,p,:] = e2p.T.reshape(shap)
        e3N[idx_below,:,p,:] = e3p.T.reshape(shap)
        _, _, _, _, _, _, d1p, d2p, d3p, e1p, e2p, e3p = d.apex_obj.basevectors_apex(GLATDS[idx_below,:,p].flatten(), 
                                                                                     GLONDS[idx_below,:,p].flatten(), alt[p]/1e3, coords='geo')
        d1S[idx_below,:,p,:] = d1p.T.reshape(shap)
        d2S[idx_below,:,p,:] = d2p.T.reshape(shap)
        d3S[idx_below,:,p,:] = d3p.T.reshape(shap)
        e1S[idx_below,:,p,:] = e1p.T.reshape(shap)
        e2S[idx_below,:,p,:] = e2p.T.reshape(shap)
        e3S[idx_below,:,p,:] = e3p.T.reshape(shap)  

    d['d1N'] = (['mlat','mlon','alt','vec'], d1N)
    d['d1S'] = (['mlat','mlon','alt','vec'], d1S)
    d['d2N'] = (['mlat','mlon','alt','vec'], d2N)
    d['d2S'] = (['mlat','mlon','alt','vec'], d2S)
    d['d3N'] = (['mlat','mlon','alt','vec'], d3N)
    d['d3S'] = (['mlat','mlon','alt','vec'], d3S)
    d['e1N'] = (['mlat','mlon','alt','vec'], e1N)
    d['e1S'] = (['mlat','mlon','alt','vec'], e1S)
    d['e2N'] = (['mlat','mlon','alt','vec'], e2N)
    d['e2S'] = (['mlat','mlon','alt','vec'], e2S)
    d['e3N'] = (['mlat','mlon','alt','vec'], e3N)
    d['e3S'] = (['mlat','mlon','alt','vec'], e3S)

    ######## Compute step sizes #########
    for m in range(M):
        printerase('WGS84 to ECEF: m = %02i / %i' % (m+1, M))
        for n in range(N):
            for p in range(P):
                lat, lon = GLATDN[m,n,p], GLONDN[m,n,p]
                if np.isfinite(lat): # Convert to ECEF
                    ECEFN[m,n,p,:] = 1e3*wgs84_to_ecef([lat, lon, alt[p]/1e3]) # [m]
                lat, lon = GLATDS[m,n,p], GLONDS[m,n,p]
                if np.isfinite(lat): # Convert to ECEF
                    ECEFS[m,n,p,:] = 1e3*wgs84_to_ecef([lat, lon, alt[p]/1e3]) # [m]

    # For now, skip lowest point since the step size can't be computed elegantly. This will be approximated below, 
    # because it is needed for the boundary condition.
    # Ultimately the better solution might be a ghost row or "sponge layer" but I don't think this makes a significant difference in the final result.
    for m in range(1,M): 
        for n in range(N):
            # Find step size for each element along B
            # Take care with the first and last point, for which the step sizes are halved.
            P1 = min(P,m+1) # For "closed" field lines, there are m+1 valid points (by construction of the grid above). 
                            # For "open" field lines (not really open, but with apex above top altitude), there are P valid points
                            # WARNING: if grid changes, this will have to change
            dxyzN = ECEFN[m,n,1:P1,:] - ECEFN[m,n,0:P1-1,:] # [m] vector from lower altitude to higher altitude. Note km to m.
            dxyzS = ECEFS[m,n,1:P1,:] - ECEFS[m,n,0:P1-1,:]
            distN = np.linalg.norm(dxyzN, axis=1) # [m] distances from point p to the next point p+1
            distS = np.linalg.norm(dxyzS, axis=1)
            # First and last point are special cases in Riemann sum. Integral is from alt[0] to alt[P1-1].
            dsN[m,n,0] = distN[0]/2.
            dsS[m,n,0] = distS[0]/2.
            dsN[m,n,P1-1] = distN[P1-2]/2.
            dsS[m,n,P1-1] = distS[P1-2]/2.
            dsN[m,n,1:P1-1] = (distN[0:P1-2] + distN[1:P1-1])/2.
            dsS[m,n,1:P1-1] = (distS[0:P1-2] + distS[1:P1-1])/2.

    # Fill in lowest step size using the one directly above it. (It's approximately constant with altitude)
    dsN[0,:,0] = dsN[1,:,1] # Note that the vertical step size is the diagonal entry, as it's stored here
    dsS[0,:,0] = dsS[1,:,1] # Note that the vertical step size is the diagonal entry, as it's stored here

    d['ECEFN'] = (['mlat','mlon','alt','vec'], ECEFN)
    d['ECEFS'] = (['mlat','mlon','alt','vec'], ECEFS)
    d['dsN'] = (['mlat','mlon','alt'], dsN)
    d['dsS'] = (['mlat','mlon','alt'], dsS)

    tstop = datetime.now()
    printerase('Total time elapsed: %.2f min' % ((tstop-tstart).total_seconds()/60.))
    
    return d


def copy(d):
    '''
    Return a deep copy of the standard Dataset d.
    
    This special function was necessary since xarray.Dataset.copy() failed due to apex_obj attribute
    '''
    
    d2 = d.copy()
    del d2.attrs['apex_obj']
    d3 = d2.copy(deep=True)
    d3.attrs['apex_obj'] = d.apex_obj
    return d3



# The interface for these two functions got a little awkward when I switched to xarray
def mlatd_to_apexht(d, mlatd):
    '''
    Given a modified apex latitude in deg, return the apex height in m
    '''
    return (d.RE+d.hR)/np.cos(mlatd*np.pi/180.)**2 - d.RE


def apexht_to_mlatd(d, h):
    '''
    Given an apex height in m, return the modified apex latitude
    '''
    return 180./np.pi * np.arccos(np.sqrt((d.RE+d.hR)/(d.RE+h)))



def get_B_IGRF(d):
    '''
    Run IGRF for each point in the grid and fill in the Be3 variables
    '''
    mlond = d.mlond
    mlatd = d.mlatd
    hR = d.hR
    M = d.M
    N = d.N
    tref = d.tref

    Be3N = np.nan * np.zeros((M,N)) # Mag field strength at hR [T]. By construction this term is constant along field lines.
    Be3S = np.nan * np.zeros((M,N))

    for m in range(M):
        for n in range(N):
            # Compute B, wind, conductivity, etc, for this point. Once for N, once for S
            lat, lon = d.apex_obj.convert( mlatd[m], mlond[n], 'apex', 'geo', height=hR/1e3)
            pt = pyglow.Point(tref, lat, lon, hR/1e3, user_ind=True)
            pt.run_igrf()
            Be3N[m,n] = np.sqrt(pt.Bx**2 +  pt.By**2 + pt.Bz**2)

            lat, lon = d.apex_obj.convert(-mlatd[m], mlond[n], 'apex', 'geo', height=hR/1e3)
            pt = pyglow.Point(tref, lat, lon, hR/1e3, user_ind=True)
            pt.run_igrf()
            Be3S[m,n] = np.sqrt(pt.Bx**2 +  pt.By**2 + pt.Bz**2)

    d['Be3N'] = (['mlat','mlon'], Be3N)
    d['Be3S'] = (['mlat','mlon'], Be3S)



def convert_winds_to_mag_coords(d):
    '''
    Rotate the wind vector from geographic to magnetic (modified apex) coordinates, and save in variables ue1N, ue2N, ue1S, ue2S
    '''

    # ue1 = d1 dot ug
    # ue2 = d2 dot ug
    ue1N = d.d1N[:,:,:,0]*d.ugN[:,:,:,0] + d.d1N[:,:,:,1]*d.ugN[:,:,:,1] # Note vertical wind is omitted
    ue2N = d.d2N[:,:,:,0]*d.ugN[:,:,:,0] + d.d2N[:,:,:,1]*d.ugN[:,:,:,1] 
    ue1S = d.d1S[:,:,:,0]*d.ugS[:,:,:,0] + d.d1S[:,:,:,1]*d.ugS[:,:,:,1]
    ue2S = d.d2S[:,:,:,0]*d.ugS[:,:,:,0] + d.d2S[:,:,:,1]*d.ugS[:,:,:,1]     
    
    d['ue1N'] = ue1N
    d['ue1S'] = ue1S
    d['ue2N'] = ue2N
    d['ue2S'] = ue2S


def compute_FLI(d):
    '''
    Perform field-line integration to get 2D variables needed for solution
    '''    

    # First, convert winds into magnetic coordinates (assuming this hasn't been done already)
    convert_winds_to_mag_coords(d)

    d1N = d.d1N.values
    d1S = d.d1S.values
    d2N = d.d2N.values
    d2S = d.d2S.values
    d3N = d.d3N.values
    d3S = d.d3S.values
    sigPN = d.sigPN.values
    sigPS = d.sigPS.values
    sigHN = d.sigHN.values
    sigHS = d.sigHS.values
    dsN = d.dsN.values
    dsS = d.dsS.values
    Be3N = d.Be3N.values
    Be3S = d.Be3S.values
    ue1N = d.ue1N.values
    ue1S = d.ue1S.values
    ue2N = d.ue2N.values
    ue2S = d.ue2S.values

    M = d.M
    N = d.N
    P = d.P


    ### Perform integrals, and combine hemispheres
    # Compute variables which are needed inside integrals
    # Recall that these terms are defined at the midpoint of the grid.
    sinIm = 2*np.sin(d.mlat)*(4-3*np.cos(d.mlat)**2)**(-0.5)
    sinIm = np.broadcast_to(sinIm, (N,M)).T # Make it a 2D array for easier coding (M,N)
    DN = np.linalg.norm(np.cross(d1N, d2N), axis=3)
    DS = np.linalg.norm(np.cross(d1S, d2S), axis=3)
    d1sq_DN = np.linalg.norm(d1N, axis=3)**2 / DN # d1^2 / D
    d1sq_DS = np.linalg.norm(d1S, axis=3)**2 / DS
    d2sq_DN = np.linalg.norm(d2N, axis=3)**2 / DN # d2^2 / D
    d2sq_DS = np.linalg.norm(d2S, axis=3)**2 / DS
    d1d2_DN  = (d1N[:,:,:,0]*d2N[:,:,:,0] + d1N[:,:,:,1]*d2N[:,:,:,1] + d1N[:,:,:,2]*d2N[:,:,:,2])/DN # d1 dot d2 / D. There is probably a faster way to do this.
    d1d2_DS  = (d1S[:,:,:,0]*d2S[:,:,:,0] + d1S[:,:,:,1]*d2S[:,:,:,1] + d1S[:,:,:,2]*d2S[:,:,:,2])/DS 

    # Sig_phi_phi
    SppN = abs(sinIm) * np.nansum(sigPN * d1sq_DN * dsN, axis=2)
    SppS = abs(sinIm) * np.nansum(sigPS * d1sq_DS * dsS, axis=2)

    # Sig_lamlam
    SllN = 1/abs(sinIm) * np.nansum(sigPN * d2sq_DN * dsN, axis=2)
    SllS = 1/abs(sinIm) * np.nansum(sigPS * d2sq_DS * dsS, axis=2)

    # Sig_H
    SHN = np.nansum(sigHN * dsN, axis=2)
    SHS = np.nansum(sigHS * dsS, axis=2)

    # Sig_C
    SCN = np.nansum(sigPN * d1d2_DN * dsN, axis=2)
    SCS = np.nansum(sigPS * d1d2_DS * dsS, axis=2)

    # Sig_lam_phi and Sig_phi_lam
    SplN =  (SHN - SCN)
    SplS = -(SHS - SCS)
    SlpN = -(SHN + SCN)
    SlpS =  (SHS + SCS)

    # K_m,phi^D. Zonal wind-driven current term.
    KpDN = Be3N * abs(sinIm) * np.nansum( (sigPN*d1sq_DN*ue2N + (sigHN - sigPN*d1d2_DN)*ue1N) * dsN, axis=2 )
    KpDS = Be3S * abs(sinIm) * np.nansum( (sigPS*d1sq_DS*ue2S + (sigHS - sigPS*d1d2_DS)*ue1S) * dsS, axis=2 )

    # K_m,lam^D. Merid wind-driven current term
    KlDN = -Be3N * np.nansum( ((sigHN + sigPN*d1d2_DN)*ue2N - sigPN*d2sq_DN*ue1N) * dsN, axis=2 )
    KlDS =  Be3S * np.nansum( ((sigHS + sigPS*d1d2_DS)*ue2S - sigPS*d2sq_DS*ue1S) * dsS, axis=2 )

    # Combine hemispheres and save
    # TODO: worth saving any more variables?
    d['KpDN'] = (['mlat','mlon'], KpDN)
    d['KpDS'] = (['mlat','mlon'], KpDS)
    d['KlDN'] = (['mlat','mlon'], KlDN)
    d['KlDS'] = (['mlat','mlon'], KlDS)
    d['Spp']  = (['mlat','mlon'], SppN + SppS)
    d['Spl']  = (['mlat','mlon'], SplN - SplS)
    d['Slp']  = (['mlat','mlon'], SlpN - SlpS)
    d['Sll']  = (['mlat','mlon'], SllN + SllS)
    d['KpD']  = (['mlat','mlon'], KpDN + KpDS)
    d['KlD']  = (['mlat','mlon'], KlDN - KlDS)
    d['sinIm'] = (['mlat','mlon'], sinIm)




def solve(d, hlb=None, llb='Kl=0', verbose=False):
    '''
    Solve the dynamo equation for potential, given the boundary conditions:

    - hlb: High latitude (i.e., high-apex-height) boundary condition, either a Dirichlet or Neumeann boundary condition
            * 'Kl=0': Neumann boundary condition to enforce no net meridional current. Numerically this will be saved as NaN.
            * (scalar or array(N)): Dirichlet boundary condition: Set Phi = this value on the boundary (can be function of mlon)
            * None: Use the "hlb" variable of d (assumed to be a function of mlon) which has already been filled in.
            * If the array is all NaNs, it will be interpreted as 'Kl=0'. This is messy but makes sense for batch jobs/recording.
    - llb: Low latitude (i.e., low-apex-height) boundary condition. Same as above:
            * 'Kl=0': Neumann boundary condition to enforce no net meridional current. Numerically this will be saved as NaN.
            * (scalar or array(N)): Dirichlet boundary condition: Set Phi = this value on the boundary (can be function of mlon)
            * None: Use the "llb" variable of d (assumed to be a function of mlon) which has already been filled in.
            * If the array is all NaNs, it will be interpreted as 'Kl=0'. This is messy but makes sense for batch jobs/recording.
    '''

    ### Handle inputs for boundary conditions 
    if hlb is None:
        assert 'hlb' in d
        hlb = d.hlb.values
    if llb is None:
        assert 'llb' in d
        llb = d.llb.values
        
    import numbers  
    if isinstance(hlb, numbers.Number):
        hlb = hlb * np.ones(d.N)    
    if isinstance(llb, numbers.Number):
        hlb = llb * np.ones(d.N)
        
    try:
        if all(np.isnan(hlb)):
            hlb = 'Kl=0'    
    except: 
        pass
    try:
        if all(np.isnan(llb)):
            llb = 'Kl=0'
    except: 
        pass
        

    Spp = d.Spp.values # Taking values here just to be 100% sure the older code below will work exactly the same as before
    Spl = d.Spl.values
    Slp = d.Slp.values
    Sll = d.Sll.values
    KpD = d.KpD.values 
    KlD = d.KlD.values
    M = d.M
    N = d.N
    P = d.P   
    d1N = d.d1N.values
    d1S = d.d1S.values
    d2N = d.d2N.values
    d2S = d.d2S.values
    d3N = d.d3N.values
    d3S = d.d3S.values
    Be3N = d.Be3N.values
    Be3S = d.Be3S.values

    mlat = d.mlat.values
    mlate = d.mlate.values
    mlon = d.mlon.values
    mlone = d.mlone.values
    sinIm = d.sinIm.values

    ## Discretize PDE and create matrix equation
    # Use sparse matrices since there are lots of zeros. See notes (which were on paper but hopefully copied to Evernote)
    # H*Phi = f where:
    #   Phi is unknown potential, organized into a 1D array
    #   f   is the the right-hand side of the dynamo equation (representing wind-driven current terms)
    #   H   is the (sparse) matrix that relates the two.
    # The high-mlat and low-mlat boundaries of Phi will need to be set by boundary conditions and do not have equations written for them.
    # Write one dynamo equation for every point on the grid *edges* (except boundaries)

    # Use np.ravel, np.reshape, np.ravel_multi_index, np.unravel_index to go back and forth between single-index and 2D index

    # (M+1)*N equations in total. There are done in this order:
    #   (M-1)*N dynamo equations
    #   N lower boundary conditions
    #   N upper boundary conditions
    # This means that the H matrix is (M+1)*N x (M+1)*N

    ############ Compute terms that are used in the dynamo equation ###########
    # Take care with which are defined on midpoints and which on edges.

    # Differential terms
    dl = mlate[1:] - mlate[:-1] # Defined on grid midpoints
    dle = np.nan * np.zeros(M+1) # Defined on grid edges
    dle[1:-1] = mlat[1:] - mlat[:-1]
    dp = mlon[1] - mlon[0] # This is set to be constant, as long as the assertion below doesn't fail
    assert np.std(np.diff(mlon)) < 1e-6

    # Interpolation terms. Currently this assumes the lon grid is regular, so this complication is only introduced for the lat grid
    # For interpolating from edges to midpoints, simply average
    # For interpolating from midpoints to edges, use a linear interpolation
    w = np.nan * np.zeros(M+1) # weighting function defined on edges, for determining an edge point
    for m in range(1,M): # No need to define grid boundaries
        w[m] = (mlate[m] - mlat[m-1])/(mlat[m] - mlat[m-1]) # see notes
        # ye[m] = (1-w)*y[m-1] + w*y[m] # formula to move from midpoint grid to edge grid

    # Conductivity terms and gradients
    Sppe = np.zeros((M+1,N)) # edges
    Sppx = (1-w[1:-1,None])*Spp[:-1,:] + w[1:-1,None]*Spp[1:,:]
    Sppe[1:-1,:] = (Sppx + np.roll(Sppx, axis=1, shift=1))/2. # Average over mlon: the point with the point before it

    Slle = np.zeros((M+1,N)) # edges
    Sllx = (1-w[1:-1,None])*Sll[:-1,:] + w[1:-1,None]*Sll[1:,:]
    Slle[1:-1,:] = (Sllx + np.roll(Sllx, axis=1, shift=1))/2. # Average over mlon: the point with the point before it
    Slle[0,:] = Sll[0,:] # This extrapolation is only necessary for boundary condition. TODO: Not any more; delete this.
    Slle[M,:] = Sll[M-1,:]

    Sple = np.zeros((M+1,N)) # edges
    Splx = (1-w[1:-1,None])*Spl[:-1,:] + w[1:-1,None]*Spl[1:,:]
    Sple[1:-1,:] = (Splx + np.roll(Splx, axis=1, shift=1))/2. # Average over mlon: the point with the point before it

    Slpe = np.zeros((M+1,N)) # edges
    Slpx = (1-w[1:-1,None])*Slp[:-1,:] + w[1:-1,None]*Slp[1:,:]
    Slpe[1:-1,:] = (Slpx + np.roll(Slpx, axis=1, shift=1))/2. # Average over mlon: the point with the point before it
    Slpe[0,:] = Slp[0,:] # Extrapolate with zero-order hold. (This is only needed for vertical-current boundary condition TODO: Not any more; delete this.
    Slpe[M,:] = Slp[M-1,:]

    dSpp_dpe = np.nan * np.zeros((M+1,N)) # partial(Sig_phi_phi)/partial(phi) defined on edges
    Sppx = (1-w[1:-1,None])*Spp[:-1,:] + w[1:-1,None]*Spp[1:,:]
    dSpp_dpe[1:-1,:] = (Sppx - np.roll(Sppx, axis=1, shift=1))/dp # Upper and lower boundaries are undefined -- they are not needed

    dSpl_dpe = np.nan * np.zeros((M+1,N)) # partial(Sig_phi_lam)/partial(phi) defined on edges
    Splx = (1-w[1:-1,None])*Spl[:-1,:] + w[1:-1,None]*Spl[1:,:]
    dSpl_dpe[1:-1,:] = (Splx - np.roll(Splx, axis=1, shift=1))/dp # Upper and lower boundaries are undefined -- they are not needed

    dSlp_dle = np.nan * np.zeros((M+1,N)) # partial(Sig_lam_phi)/partial(lam) defined on edges
    Slpx = (Slp + np.roll(Slp, axis=1, shift=1))/2. # Average over mlon: the point with the point before it
    dSlp_dle[1:-1,:] = (Slpx[1:,:] - Slpx[:-1,:])/dle[1:-1,None] # Upper and lower boundaries are undefined -- they are not needed

    dSllcosl_dle = np.nan * np.zeros((M+1,N)) # partial(Sig_lam_lam*cos(lam))/partial(lam) defined on edges
    Sllxcosl = np.cos(mlat)[:,None] * (Sll + np.roll(Sll, axis=1, shift=1))/2. # Average over mlon: the point with the point before it
    dSllcosl_dle[1:-1,:] = (Sllxcosl[1:,:] - Sllxcosl[:-1,:])/dle[1:-1,None] # Upper and lower boundaries are undefined -- they are not needed

    # Need to define terms on phi-grid midpoints if the Neumann boundary condition is used.
    Slp0 = (Slp + np.roll(Slp, axis=1, shift=1))/2.
    Sll0 = (Sll + np.roll(Sll, axis=1, shift=1))/2.
    KlD0 = (KlD + np.roll(KlD, axis=1, shift=1))/2.

    cosle = np.cos(mlate) # cos(lam) defined on edges

    ## Construct matrix sparsely. Each entry of these vectors is an entry in the matrix.
    Hi = [] # which row
    Hj = [] # which column
    Hv = [] # what the value is

    ############ Dynamo equations #############
    for m in range(1,M): # all mlats except edges
        for n in range(N):
            # Record dynamo equation for mlate[m], mlone[n]
            i = (m-1)*N + n # this counts which equation number (i.e., which row of H)

            # Phi[m,n] terms
            j = np.ravel_multi_index((m,n), (M+1,N))
            v = -2*Sppe[m,n]/(dp**2 * cosle[m]) - Slle[m,n]*cosle[m]*(1/(dl[m]*dle[m]) + 1/(dl[m-1]*dle[m]))
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)

            # Phi[m,n+1] terms
            j = np.ravel_multi_index((m,np.mod(n+1,N)), (M+1,N)) # Note mod to allow lon wraparound
            v =  1/(2*dp)*dSpp_dpe[m,n]/cosle[m] + Sppe[m,n]/(dp**2 * cosle[m]) + 1/(2*dp) * dSlp_dle[m,n]
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)

            # Phi[m,n-1] terms
            j = np.ravel_multi_index((m,np.mod(n-1,N)), (M+1,N)) # Note mod to allow lon wraparound
            v = -1/(2*dp)*dSpp_dpe[m,n]/cosle[m] + Sppe[m,n]/(dp**2 * cosle[m]) - 1/(2*dp) * dSlp_dle[m,n]
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)

            # Phi[m+1,n] terms
            j = np.ravel_multi_index((m+1,n), (M+1,N))
            v =  1/(2*dle[m])*dSpl_dpe[m,n] + 1/(2*dle[m])*dSllcosl_dle[m,n] + Slle[m,n]*cosle[m]/(dl[m]*dle[m])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)

            # Phi[m-1,n] terms
            j = np.ravel_multi_index((m-1,n), (M+1,N))
            v = -1/(2*dle[m])*dSpl_dpe[m,n] - 1/(2*dle[m])*dSllcosl_dle[m,n] + Slle[m,n]*cosle[m]/(dl[m-1]*dle[m])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)

            # Phi[m+1,n+1]
            j = np.ravel_multi_index((m+1,np.mod(n+1,N)), (M+1,N))
            v =  (Sple[m,n] + Slpe[m,n])/(4*dp*dle[m])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)

            # Phi[m-1,n+1]
            j = np.ravel_multi_index((m-1,np.mod(n+1,N)), (M+1,N))
            v = -(Sple[m,n] + Slpe[m,n])/(4*dp*dle[m])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)

            # Phi[m+1,n-1]
            j = np.ravel_multi_index((m+1,np.mod(n-1,N)), (M+1,N))
            v = -(Sple[m,n] + Slpe[m,n])/(4*dp*dle[m])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)

            # Phi[m-1,n-1]
            j = np.ravel_multi_index((m-1,np.mod(n-1,N)), (M+1,N))
            v =  (Sple[m,n] + Slpe[m,n])/(4*dp*dle[m])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)


    ##### RHS
    f = np.nan * np.zeros((M+1)*N) # right hand side of the full equation, including dynamo equations and boundary conditions

    ## Dynamo equation RHS
    # Use first-order differences for RHS. Note that averaging is needed in the other dimension to define it on the edges
    KpDx = (1-w[1:-1,None])*KpD[:-1,:] + w[1:-1,None]*KpD[1:,:] # Interpolate over mlat
    dKpD_dp = (KpDx - np.roll(KpDx, axis=1, shift=1))/dp # Derivative in mlon. Point and the point before it, to define on the edges
                                                         # Note the mlat boundaries are omitted here.

    KlDxcosl = np.cos(mlat)[:,None]*(KlD + np.roll(KlD, axis=1, shift=1))/2. # Average over mlon
    dKlDcosl_dl = (KlDxcosl[1:,:] - KlDxcosl[:-1,:])/dle[1:-1,None] # Note the mlat boundaries are omitted here.

    dynamo_rhs = (d.RE+d.hR) * (dKpD_dp + dKlDcosl_dl) 
    f[:(M-1)*N] = dynamo_rhs.ravel()


    ########### Boundary conditions ###########
    #### Upper boundary
    if isinstance(hlb,str) and np.array_equal(hlb, 'Kl=0'): # Is this the best way to do str equals to avoid deprecation warning?
        for n in range(N):
            i = (M-1)*N + n # this counts which equation number (i.e., which row of H)      

            # Vertical current is 0. This eqn is written at mlat[M-1], mlone[n] (i.e., on the phi-grid edge and lambda-grid midpoint)

            # Phi[M-1,n] terms
            j = np.ravel_multi_index((M-1,np.mod(n,N)), (M+1,N))
            v = -1/(dl[M-1])*Sll0[M-1,n] - 1/(4*dp)*Slp0[M-1,n]/np.cos(mlat[M-1])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v) 

            # Phi[M,n] terms
            j = np.ravel_multi_index((M,np.mod(n,N)), (M+1,N))
            v = 1/(dl[M-1])*Sll0[M-1,n] - 1/(4*dp)*Slp0[M-1,n]/np.cos(mlat[M-1])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v) 

            # Phi[M-1,n+1] terms
            j = np.ravel_multi_index((M-1,np.mod(n+1,N)), (M+1,N))
            v = 1/(4*dp)*Slp0[M-1,n]/np.cos(mlat[M-1])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v) 

            # Phi[M,n+1] terms
            j = np.ravel_multi_index((M,np.mod(n+1,N)), (M+1,N))
            v = 1/(4*dp)*Slp0[M-1,n]/np.cos(mlat[M-1])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v) 

        # RHS
        f[(M-1)*N:M*N] = (RE+hR) * KlD0[-1,:]

    else: # Use Dirichlet boundary condition
        for n in range(N):
            i = (M-1)*N + n # this counts which equation number (i.e., which row of H)

            # set to a given value
            j = np.ravel_multi_index((M,n), (M+1,N))
            v = 1
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)       
        # RHS
        f[(M-1)*N:M*N] = hlb

    #### Lower boundary
    if isinstance(llb,str) and np.array_equal(llb, 'Kl=0'): # Is this the best way to do str equals to avoid deprecation warning?
        for n in range(N):
            i = M*N + n # this counts which equation number (i.e., which row of H)

            # Vertical current is 0. This eqn is written at mlat[0], mlon[n] (i.e., on the phi-grid edge and lambda-grid midpoint)

#                 # Phi[0,n] terms
#                 j = np.ravel_multi_index((0,np.mod(n,N)), (M+1,N))
#                 v = -1/(dl[0])*Sll0[0,n] -1/(4*dp)*Slp0[0,n]/np.cos(mlat[0])
#                 Hi.append(i)
#                 Hj.append(j)
#                 Hv.append(v) 

#                 # Phi[1,n] terms
#                 j = np.ravel_multi_index((1,np.mod(n,N)), (M+1,N))
#                 v = 1/(dl[0])*Sll0[0,n] - 1/(4*dp)*Slp0[0,n]/np.cos(mlat[0])
#                 Hi.append(i)
#                 Hj.append(j)
#                 Hv.append(v) 

#                 # Phi[0,n+1] terms
#                 j = np.ravel_multi_index((0,np.mod(n+1,N)), (M+1,N))
#                 v = 1/(4*dp)*Slp0[0,n]/np.cos(mlat[0])
#                 Hi.append(i)
#                 Hj.append(j)
#                 Hv.append(v) 

#                 # Phi[1,n+1] terms
#                 j = np.ravel_multi_index((1,np.mod(n+1,N)), (M+1,N))
#                 v = 1/(4*dp)*Slp0[0,n]/np.cos(mlat[0])
#                 Hi.append(i)
#                 Hj.append(j)
#                 Hv.append(v) 

            # Phi[0,n] terms
            j = np.ravel_multi_index((0,np.mod(n,N)), (M+1,N))
            v = -1/(dl[0])*Sll[0,n] -1/dp*Slp[0,n]/np.cos(mlat[0])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v) 

            # Phi[1,n] terms
            j = np.ravel_multi_index((1,np.mod(n,N)), (M+1,N))
            v = 1/(dl[0])*Sll[0,n]
            Hi.append(i)
            Hj.append(j)
            Hv.append(v) 

            # Phi[0,n+1] terms
            j = np.ravel_multi_index((0,np.mod(n+1,N)), (M+1,N))
            v = 1/dp*Slp[0,n]/np.cos(mlat[0])
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)

        # RHS
        f[M*N:(M+1)*N] = (d.RE+d.hR) * KlD[0,:] # Option 2: edge-midpoint with forward difference 

    else: # Use Dirichlet boundary condition
        for n in range(N):
            i = M*N + n # this counts which equation number (i.e., which row of H)

            # set to a given value
            j = np.ravel_multi_index((0,n), (M+1,N))
            v = 1
            Hi.append(i)
            Hj.append(j)
            Hv.append(v)       
        # RHS
        f[M*N:(M+1)*N] = llb



    ######### Solve ##########

    H = sp.coo_matrix((Hv,[Hi,Hj]), shape=((M+1)*N, (M+1)*N))
    Phi1D = sp.linalg.spsolve(H.tocsc(), f)
    Phi = np.reshape(Phi1D, (M+1,N))
    resid = H@Phi1D - f
    if verbose:
        print('If this is small, a solution was found: %e' % (np.linalg.norm(resid)/np.linalg.norm(f)))

    # More physical quantities
    # Electric fields
    Phix = (Phi[1:,:] + Phi[:-1,:])/2. # Average over mlat to define on midpoints.
    Ep = -1/((d.RE+d.hR)*np.cos(mlat[:,None])) * 1/dp * (np.roll(Phix,shift=-1,axis=1) - Phix) # forward difference
    Ed1 = Ep
    Phix = (Phi + np.roll(Phi, axis=1, shift=-1))/2. # Average over mlon: the point with the point after it (to define on midpoints)
    El = -1/(d.RE+d.hR) * 1/dl[:,None] * (Phix[1:,:] - Phix[:-1,:])
    Ed2 = -El/sinIm

    # ExB drifts mapped to reference height
    ve1N =  Ed2/Be3N
    ve1S =  Ed2/Be3S
    ve2N = -Ed1/Be3N
    ve2S = -Ed1/Be3S

    # Currents
    DN = np.linalg.norm(np.cross(d1N, d2N), axis=3)
    DS = np.linalg.norm(np.cross(d1S, d2S), axis=3)
    d1sqN = np.linalg.norm(d1N, axis=3)**2
    d1sqS = np.linalg.norm(d1S, axis=3)**2
    d2sqN = np.linalg.norm(d2N, axis=3)**2
    d2sqS = np.linalg.norm(d2S, axis=3)**2
    d1d2N  = (d1N[:,:,:,0]*d2N[:,:,:,0] + d1N[:,:,:,1]*d2N[:,:,:,1] + d1N[:,:,:,2]*d2N[:,:,:,2])
    d1d2S  = (d1S[:,:,:,0]*d2S[:,:,:,0] + d1S[:,:,:,1]*d2S[:,:,:,1] + d1S[:,:,:,2]*d2S[:,:,:,2])

    # Richmond (1995) Eq 5.7, just the wind terms
    je1DN = d.sigPN * d1sqN * d.ue2N * Be3N[:,:,None] + (d.sigPN * d1d2N - d.sigHN * DN)* -d.ue1N * Be3N[:,:,None]  
    je1DS = d.sigPS * d1sqS * d.ue2S * Be3S[:,:,None] + (d.sigPS * d1d2S - d.sigHS * DS)* -d.ue1S * Be3S[:,:,None]  
    # Richmond (1995) Eq 5.7, just the E-field terms
    je1EN = d.sigPN * d1sqN * Ed1[:,:,None] + (d.sigPN * d1d2N - d.sigHN * DN)*  Ed2[:,:,None]
    je1ES = d.sigPS * d1sqS * Ed1[:,:,None] + (d.sigPS * d1d2S - d.sigHS * DS)*  Ed2[:,:,None]
    # Richmond (1995) Eq 5.8, just the wind terms
    je2DN = (d.sigPN * d1d2N + d.sigHN * DN) * d.ue2N * Be3N[:,:,None] + d.sigPN * d2sqN * -d.ue1N * Be3N[:,:,None]
    je2DS = (d.sigPS * d1d2S + d.sigHS * DS) * d.ue2S * Be3S[:,:,None] + d.sigPS * d2sqS * -d.ue1S * Be3S[:,:,None]
    # Richmond (1995) Eq 5.8, just the E-field terms
    je2EN = (d.sigPN * d1d2N + d.sigHN * DN) * Ed1[:,:,None] + d.sigPN * d2sqN *  Ed2[:,:,None]   
    je2ES = (d.sigPS * d1d2S + d.sigHS * DS) * Ed1[:,:,None] + d.sigPS * d2sqS *  Ed2[:,:,None]         
    je1N = je1DN + je1EN
    je1S = je1DS + je1ES
    je2N = je2DN + je2EN
    je2S = je2DS + je2ES

    ######## Save ########
    d['Phi'] = (['mlate','mlone'], Phi)
    d['resid_rms'] = np.mean(resid**2)
    d['Ed1'] = (['mlat','mlon'],Ed1)
    d['Ed2'] = (['mlat','mlon'],Ed2)
    d['El'] = (['mlat','mlon'],El)
    d['Ep'] = (['mlat','mlon'],Ep)
#     d.attrs['f'] = f # Skipping these two since they take a lot of space and don't generalize well.
#     d.attrs['H'] = H
    d['ve1N'] = (['mlat','mlon'],ve1N)
    d['ve1S'] = (['mlat','mlon'],ve1S)
    d['ve2N'] = (['mlat','mlon'],ve2N)
    d['ve2S'] = (['mlat','mlon'],ve2S)
#     d['je1N'] = (['mlat','mlon','alt'],je1N)
#     d['je1S'] = (['mlat','mlon','alt'],je1S)
#     d['je2N'] = (['mlat','mlon','alt'],je2N)
#     d['je2S'] = (['mlat','mlon','alt'],je2S)
    d['je1N'] = je1N
    d['je1S'] = je1S
    d['je2N'] = je2N
    d['je2S'] = je2S
     
    # Special case for saving/recording boundary conditions. (This is kind of messy but couldn't figure out a cleaner way)
    if isinstance(hlb,str) and np.array_equal(hlb, 'Kl=0'):
        d['hlb']  = (['mlon'], np.nan*np.zeros(N))
    else:
        d['hlb']  = (['mlon'], hlb)
    if isinstance(llb,str) and np.array_equal(llb, 'Kl=0'):
        d['llb']  = (['mlon'], np.nan*np.zeros(N))
    else:
        d['llb']  = (['mlon'], llb)

        

###################### Functions incorporating time dimension ########################



def concat_over_time(d_list):
    '''
    Concatenate a list of dynamo Datasets in the "standard" form into a new Dataset, along
    the time dimension. It's assumed that "time" is a (single-length) coordinate of each. 
    
    I was hoping to simply use xr.concat, but the "data_vars='different'" input was either
    buggy or not working the way I understood. So I'll do it manually.
    '''
    
#     # First, make sure time is a coordinate
    assert 'time' in d_list[0].coords
    # Initial concat
    d = xr.concat(d_list, dim='time', combine_attrs='drop_conflicts')
    
    # Re-write Apr 18 to manually check each variable to see if it's changing
    for v in d.data_vars:
        rel_change = d[v].std(dim='time').mean() / d[v].std()
        if rel_change < 1e-12:
            d[v] = d_list[0][v]
        
        
    # The old way, which manually listed which variables to keep constant:
#     # List of variables that are NOT time dependent. Time will not be a dimension of these.
#     const_vars = ['mlatde', 'mlatd', 'mlond', 'mlonde', 'alt', 'hAe', 'hA', 'GLATDN', 'GLATDS', 'GLONDN', 'GLONDS', 
#      'd1N', 'd1S', 'd2N', 'd2S', 'd3N', 'd3S', 'e1N', 'e1S', 'e2N', 'e2S', 'e3N', 'e3S', 'ECEFN', 'ECEFS', 'dsN', 'dsS',
#      'Be3N', 'Be3S', 'sinIm']    
#     # Pin the values to their initial value.
#     for v in const_vars:
#         if v in d.variables and 'time' in d[v].coords: # If the variable's actually in there, and if it depends on time:
#             assert d[v].std(dim='time').all() < 1e-12, "%s is changing over time" % v # Make sure it's not changing
#             d[v] = d_list[0][v]

    return d
    

def concat_over_doy(d_list):
    '''
    Concatenate a list of dynamo Datasets in the "standard" form into a new Dataset, along
    the doy dimension. It's assumed that "doy" is a (single-length) coordinate of each. 
    
    I was hoping to simply use xr.concat, but the "data_vars='different'" input was either
    buggy or not working the way I understood. So I'll do it manually.
    '''
    
#     # First, make sure time is a coordinate
    assert 'doy' in d_list[0].coords
    # Initial concat
    d = xr.concat(d_list, dim='doy', combine_attrs='drop_conflicts')
    
    # Re-write Apr 18 to manually check each variable to see if it's changing
    for v in d.data_vars:
        rel_change = d[v].std(dim='doy').mean() / d[v].std()
        if rel_change < 1e-12:
            d[v] = d_list[0][v]
    
    return d



def concat_over_drivers(d_list):
    '''
    Concatenate a list of dynamo Datasets in the "standard" form into a new Dataset, along
    the "driver" dimension (presumably consisting of cases run with different drivers). 
    It's assumed that "driver" is a (single-length) coordinate of each. 
    
    I was hoping to simply use xr.concat, but the "data_vars='different'" input was either
    buggy or not working the way I understood. So I'll do it manually.
    '''
    
#     # First, make sure time is a coordinate
    assert 'driver' in d_list[0].coords
    # Initial concat
    d = xr.concat(d_list, dim='driver', combine_attrs='drop_conflicts')
    
    # Re-write Apr 18 to manually check each variable to see if it's changing
    for v in d.data_vars:
        rel_change = d[v].std(dim='driver').mean() / d[v].std()
        if rel_change < 1e-12:
            d[v] = d_list[0][v]
    
    return d
    
    
    
def view_by_slt(d, N=100):
    '''
    Create a copy of this Dynamo object, re-indexing the mlon dimension to LT, and shift/interpolate the data
    to make that happen. This interpolates *both* midpoint (mlon) and edge (mlone) to a common LT grid.
    
    Note that this just does mlon, not mlone (so Phi is not yet included). This would
    be trivial to include if needed.
    
    N = number of samples of LT
    
    This masks the variable "slteq" because it would need special treatment and isn't needed.

    Note: Uses SLT from the "slteq" variable calculated above, which uses the equation of time
    TODO: See if I can make this more memory efficient. N=100 uses about 5GB for some reason. It 
          blows up at the concat step. I could also only keep the important variables here.
          But note that you can likely just delete this dataset after it's used.
    '''

    # Compute LT grid. It's nonregular, but as long as N=24,48,72,etc., then it will work
    # _, lon_at_mageq_at_hR = d.apex_obj.convert(0, d.mlond, 'apex', 'geo', height=d.hR/1e3, )
    # sltg = sorted(np.mod(lon_at_mageq_at_hR/15.,24.))
    sltg = np.linspace(0,24,N)[:-1]
    
    if N<len(d.mlon):
        print('Are you sure you don\'t want a finer interpolation?')
    
    def interpolated_to_slt(dn):
        _, lon_at_mageq_at_hR  = dn.apex_obj.convert(0, dn.mlond,  'apex', 'geo', height=dn.hR/1e3, )
        _, lone_at_mageq_at_hR = dn.apex_obj.convert(0, dn.mlonde, 'apex', 'geo', height=dn.hR/1e3, )
        t = pd.to_datetime(dn.time.item())
        dn['slte'] = (['mlone'], np.array([compute_slt(t, lon) for lon in lone_at_mageq_at_hR]))
        dn['slt']  = (['mlon'],  np.array([compute_slt(t, lon) for lon in lon_at_mageq_at_hR]))
        dn = dn.swap_dims({'mlon':'slt', 'mlone':'slte'}).sortby(['slt','slte'])
        
        ### This chunk of code makes sure there is edge protection for the slt and slte variables.
        vars_slt  = [v for v in dn.data_vars if 'slt' in  dn[v].dims]
        vars_slte = [v for v in dn.data_vars if 'slte' in dn[v].dims]
        vars_other = [v for v in dn.data_vars if v not in vars_slt and v not in vars_slte]
        
        dv = dn[vars_slt]
        dv0 = dv.isel(slt=-1).assign_coords(slt=dv.slt[-1]-24)
        dv1 = dv.isel(slt=0).assign_coords(slt=dv.slt[0]+24)
        dvv_slt = xr.concat([dv0, dv, dv1], dim='slt')

        dv = dn[vars_slte]
        dv0 = dv.isel(slte=-1).assign_coords(slte=dv.slte[-1]-24)
        dv1 = dv.isel(slte=0).assign_coords(slte=dv.slte[0]+24)
        dvv_slte = xr.concat([dv0, dv, dv1], dim='slte')

        dn = xr.merge([dvv_slt, dvv_slte, dn[vars_other]])
        dn['slteq'] = np.nan*dn['slteq'] # "Erase" this variable which we know has issues.
        ### End chunk
        
        # For periodic variables, take care with interpolation
        vlist = ['mlon','mlone']
        for v in vlist:
            dn[v+'_x'] = np.cos(dn[v])
            dn[v+'_y'] = np.sin(dn[v])
        
        # Do the interpolation
#         dni = dn.interp(slt=sltg, slte=sltg, kwargs={"fill_value": 'extrapolate'}).assign_coords(slt=sltg, slte=sltg)
        dni = dn.interp(slt=sltg, slte=sltg).assign_coords(slt=sltg, slte=sltg)
        
        # Reconstruct periodic variables
        for v in vlist:
            dni[v] = np.arctan2(dni[v+'_y'], dni[v+'_x'])
            dni = dni.drop([v+'_x', v+'_y'])
        dni['mlond'] = dni['mlon']*180./np.pi
        dni['mlonde'] = dni['mlone']*180./np.pi
    
        return dni

    # Make this work for both non-time-indexed and time-indexed Datasets. 
    if ('time' not in d.coords):
        d['time'] = d.tref
    if (d.time.ndim==0):
        return interpolated_to_slt(d)

    # For each timestamp, interpolate to grid 
    dns = []
    for n in range(len(d.time)):
        dni = interpolated_to_slt(d.isel(time=n))
        dns.append(dni)
    
    d2 = concat_over_time(dns)
    # Manually make sure apex_obj is there. It got dropped for some reason
    d2.attrs['apex_obj'] = d.apex_obj

    return d2
    

def view_by_mlon(d, mlong=None, mloneg=None):
    '''
    Take "d", a Dynamo object indexed by slt, and reindex it back to the normal mlon/mlone coordinates (making a copy). It's the opposite
    of view_by_slt)
    
    mlong - mlon grid to interpolate to. If None, uses the direct conversion from LT (this is untested).
    mlonge - same, for mlone grid (edge grid)
    '''

    dns = []
    for n in range(len(d.time)):
        dn = d.isel(time=n)
        dn = dn.swap_dims({'slt':'mlon', 'slte':'mlone'}).sortby(['mlon','mlone'])
        if (mlong is not None) and (mloneg is not None):

            ### This chunk of code makes sure there is edge protection for the mlon and mlone variables.
            vars_slt  = [v for v in dn.data_vars if 'mlon' in  dn[v].dims]
            vars_slte = [v for v in dn.data_vars if 'mlone' in dn[v].dims]
            vars_other = [v for v in dn.data_vars if v not in vars_slt and v not in vars_slte]

            dv = dn[vars_slt]
            dv0 = dv.isel(mlon=-1).assign_coords(mlon=dv.mlon[-1]-2*np.pi)
            dv1 = dv.isel(mlon=0).assign_coords(mlon=dv.mlon[0]+2*np.pi)
            dvv_slt = xr.concat([dv0, dv, dv1], dim='mlon')

            dv = dn[vars_slte]
            dv0 = dv.isel(mlone=-1).assign_coords(mlone=dv.mlone[-1]-2*np.pi)
            dv1 = dv.isel(mlone=0).assign_coords(mlone=dv.mlone[0]+2*np.pi)
            dvv_slte = xr.concat([dv0, dv, dv1], dim='mlone')

            dn = xr.merge([dvv_slt, dvv_slte, dn[vars_other]])
            ### End chunk


            dn = dn.interp(mlon=mlong, mlone=mloneg)
        dns.append(dn)
        
    d2 = concat_over_time(dns)
    return d2

        
def compute_FLI_solve_over_time(d, hlb=None, llb='Kl=0'):
    '''
    Same as "solve" except is generalized to handle a Dataset with multiple timestamps, 
    which are looped over and the result reconstructed. Note that unlike
    "solve" this actually returns a new variable (I couldn't figure out how
    to do this otherwise)
    
    RETURNS:
     - d, a copy of the input except with the dynamo problem solved (and new
          variables added)
    '''
    assert 'time' in d.coords
    
    dvec = []
    for i in range(len(d.time)):
        di = d.isel(time=i)
        compute_FLI(di)
        solve(di, verbose=False)
        dvec.append(di)
    d2 = concat_over_time(dvec)
    return d2





    
    
    
    
    
    
    
    
    
    
    
    