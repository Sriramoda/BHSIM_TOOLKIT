#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf
import yt
import pandas as pd
import os
from glob import glob
from yt.data_objects.particle_filters import add_particle_filter
from yt.frontends.ramses.io import convert_ramses_conformal_time_to_physical_age
from decimal import Decimal
from yt.utilities.cosmology import Cosmology


# In[3]:

co = Cosmology(
    hubble_constant=0.6736,
    omega_matter=0.315299987792969,
    omega_lambda=0.684700012207031,
    omega_curvature=0.0,
    omega_radiation=0.0,
)


def percentile16(data):
    return np.percentile(data,16)
def percentile84(data):
    return np.percentile(data,84)



def BH_information(sink_file,Osyris_file,ds_yt):
    #ds_yt used to convert the age of BH to physical regarding to current snapshot
    
    time_salpeter = 45e6                  # Salpeter time from (By P. Biernacki, R. Teyssier and A. Bleuler 2017)
    c_light = 3*10**10                    # light speed in cm/s
    
    #informations from sink files
    BHid = sink_file[:,0]                                # Black hole ID
    BHx  = sink_file[:,2]                                #x coordinate of BH in code length
    BHy  = sink_file[:,3]                                #y coordinate of BH in code length
    BHz  = sink_file[:,4]                                #z coordinate of BH in code length
    tf   = sink_file[:,11]                               #formation time in conformal time unit
                                                                 #(Use yt frontend to change to physical)
    
    
    
    #informations from Osyris
    BH_M = Osyris_file["sink"]["mbh"].to("M_sun").values                 # Mass of Black hole in Msun
    BH_acc = Osyris_file["sink"]["acc_rate"].to("M_sun/yr").values       # BH accretion rate  in Msun/yr
    BH_acc_cgs = Osyris_file["sink"]["acc_rate"].to("g/s").values        # BH accretion rate  in cgs
    BH_therm = (Osyris_file["sink"]["etherm"].to("cm**2*g/s**2").values)         #Thermal energy in cgs
    Sink_M = Osyris_file["sink"]["msink"].to("M_sun").values             #Sink Mass in Msun
    
    BHcs_cgs = np.sqrt(Osyris_file["sink"]["cs**2"].to("cm**2/s**2").values)  #gas sound speed cs in sink sphere
    BHcs_kms = np.sqrt(Osyris_file["sink"]["cs**2"].to("km**2/s**2").values)  #gas sound speed cs in sink sphere
    BHcs_SI = np.sqrt(Osyris_file["sink"]["cs**2"].to("m**2/s**2").values)  #gas sound speed cs in sink sphere
    
    Sink_vgx = Osyris_file["sink"]["v_gas"].x.to("km/s").values                   #Gas velocity x within sink
    Sink_vgy = Osyris_file["sink"]["v_gas"].y.to("km/s").values                   #Gas velocity y within sink
    Sink_vgz = Osyris_file["sink"]["v_gas"].z.to("km/s").values                   #Gas velocity z within sink
    
    
    #Calculating age of BHs
    Age_BH = convert_ramses_conformal_time_to_physical_age(ds_yt,tf)
    BH_age = np.array(Age_BH*ds_yt.time_unit.in_units("Gyr"))             #Age of BH in Gyr
    
    #Calculating Eddington rate and Eddington ratio  (By P. Biernacki, R. Teyssier and A. Bleuler 2017)
    eddington_rate = Sink_M/time_salpeter                               #Eddington rate of BH
    edd_ratio = BH_acc/eddington_rate                                   #Eddington ratio of BH
    
    #Calculating Mach number of Gas within sink (Mach = V_gas/C_s(sound speed))
    Sink_gv = np.sqrt(Sink_vgx**2 + Sink_vgy**2 + Sink_vgz**2)
    Mach_BH = Sink_gv/BHcs_kms
    
    #Energy output(Luminosity) of AGN  (By P. Biernacki, R. Teyssier and A. Bleuler 2017)
    L_agn = 0.15 * 0.1 * BH_acc_cgs * c_light**2 * 10**7
    
    BH_Properties = np.c_[BHid,BHx,BHy,BHz,BH_M,BH_age,BH_acc,BH_therm,edd_ratio,L_agn,BHcs_kms,Mach_BH]
    
    return BH_Properties

def BH_Galaxy_IDmatch(gal_data,BH_data):
    gal_i = []
    BH_ID = []
    BH_gal_dict = {}
    for BH in BH_data:
        x_n = gal_data[:,1] - BH[1]
        y_n = gal_data[:,2] - BH[2]
        z_n = gal_data[:,3] - BH[3]
        
        gal_id = gal_data[:,0]
 
        pos = np.c_[x_n,y_n,z_n]
        dist = np.linalg.norm(pos,axis=1)
        Rgal = gal_data[:,4]
        
        gal_id_masked = gal_id[dist<=0.1*Rgal]
        if len(gal_id_masked)!=0:
            BH_gal_dict[BH[0]] = {'BH_ID':BH[0],'Gal_ID': gal_id_masked}
    return BH_gal_dict

def halo_pairs(HP,BH_Gal_dict):
    id1=[]
    id2=[]
    Bid=[]
    Black_holes = BH_Gal_dict.keys()
    for i in Black_holes:
        SIM1_ID = HP[:,0][HP[:,0]==BH_Gal_dict[i]['Gal_ID']]
        SIM2_ID = HP[:,1][HP[:,0]==BH_Gal_dict[i]['Gal_ID']]
        
        Bid.append(i)
        id1.append(SIM1_ID)
        id2.append(SIM2_ID)
    return np.c_[id1,id2,Bid]

def halo_center(data):
    """
    halo_center function fetches x,y,z coordinates from AHF catalog for desired halos.
    Criteria for Halo selection are:
    1. Only Host halos(no substructures)
    2. Only halos that have non zero stellar mass
    3. Only halos that have more than 10 star particles
    
    Coordinate units in AHF (kpc/h) is converted into Ramses code unit by normalizing by box size (17e3 kpc/h)
    
    """
    x_c=[]
    y_c=[]
    z_c=[]
    all_halo_center = np.c_[data[:,5],data[:,6],data[:,7],data[:,0]]
    for halos in data:
        if halos[1]==-1:
            if halos[64]>0:
                if halos[63]>10:
                    x_c.append(halos[5])
                    y_c.append(halos[6])
                    z_c.append(halos[7])
    Halo_center = np.vstack(np.meshgrid((x_c,y_c,z_c))).reshape(3,-1).T
    return Halo_center



def halo_virial_radius(data):
    """
    halo_virial_radius function fetches virial radius from AHF catalog for desired halos.
    Criteria for Halo selection are:
    1. Only Host halos(no substructures)
    2. Only halos that have non zero stellar mass
    3. Only halos that have more than 10 star particles
    
    Virial radius units in AHF (kpc/h) is converted into Ramses code unit by normalizing by box size (17e3 kpc/h)
    """
    rvir=[]
    for halos in data:
        if halos[1]==-1:
            if halos[64]>0:
                if halos[63]>10:
                    rvir.append(halos[11])
    R_vir = np.array(rvir)
    return R_vir



def halo_id(data):
    """
    halo_id function fetches ID from AHF catalog for desired halos.
    Criteria for Halo selection are:
    1. Only Host halos(no substructures)
    2. Only halos that have non zero stellar mass
    3. Only halos that have more than 10 star particles

    """
    id_=[]
    for halos in data:
        if halos[1]==-1:          
            if halos[64]>0:
                if halos[63]>10:
                    id_.append(halos[0])
    ID = np.array(id_)
    return ID


def halo_mass(data):
    """
    halo_id function fetches ID from AHF catalog for desired halos.
    Criteria for Halo selection are:
    1. Only Host halos(no substructures)
    2. Only halos that have non zero stellar mass
    3. Only halos that have more than 10 star particles

    """
    H_mass=[]
    for halos in data:
        if halos[1]==-1:          
            if halos[64]>0:
                if halos[63]>10:
                    H_mass.append(halos[3])
    Halo_mass = np.array(H_mass)
    return Halo_mass


  


# In[4]:


def convergence_test(xo,yo,zo,xn,yn,zn):
    """
    convergence_test finds convergence between old and new centers from galaxy_com_center.
    This function works within galaxy com center
    """
    x_conv = np.abs(xo-xn)
    y_conv = np.abs(yo-yn)
    z_conv = np.abs(zo-zn)
    return x_conv,y_conv,z_conv


# In[14]:


def galaxy_com_center(center, factor, Rvir,id_,H_mass,strm):
    """
    galaxy_com_center function finds center of galaxy located within halo by calculating COM of
    all stellar particles that lies within 0.1 times virial radius of halo.
    
    center = Halo centers from AHF with units (kpc/h)
    
    factor = 0.1 or less, (0.1*H_radius) approximately equal to galactic radius
    
    Rvir =  Virial radius of halo from AHF with units(kpc/h)
    
    strm = array of stellar coordinates and mass with shape of [n,4] units [kpc/h,kpc/h,kpc/h,Msun]
    """
    x_c,y_c,z_c = [],[],[]
    rv   = []
    idd  = [] 
    h_m = []
    M_tot = []
    convergence_x=[]
    convergence_y=[]
    convergence_z=[]
    
    for c,r,i,m in zip(center,Rvir,id_,H_mass):
        x_n = c[0] - strm[:,0]
        y_n = c[1] - strm[:,1]
        z_n = c[2] - strm[:,2]
    
        mass_star = strm[:,3]
        star_x = strm[:,0]
        star_y = strm[:,1]
        star_z = strm[:,2]
    
        pos = np.c_[x_n,y_n,z_n] 
    
        dist = np.linalg.norm(pos,axis=1)
        Rgal = factor*r
        massgal = 0.1*r
    
        Star_inside = np.where(dist<=Rgal)
        mass_inside = np.where(dist<=massgal)
        x_i = star_x[Star_inside]
        y_i = star_y[Star_inside]
        z_i = star_z[Star_inside]
        m_i = mass_star[Star_inside]
        full_mass = mass_star[mass_inside]
        if np.sum(m_i)!=0:
            x_new = np.sum(x_i*m_i)/np.sum(m_i)
            y_new = np.sum(y_i*m_i)/np.sum(m_i)
            z_new = np.sum(z_i*m_i)/np.sum(m_i)
            x_c.append(x_new)
            y_c.append(y_new)
            z_c.append(z_new)
            M_tot.append(np.sum(full_mass))
            rv.append(r)
            idd.append(i)
            h_m.append(m)
            conv_x,conv_y,conv_z = convergence_test(c[0],c[1],c[2],x_new,y_new,z_new)
            convergence_x.append(conv_x)
            convergence_y.append(conv_y)
            convergence_z.append(conv_z)
            
        
    
    return np.array(x_c), np.array(y_c), np.array(z_c),np.array(rv),np.array(idd),np.array(M_tot),np.array(h_m),np.array(convergence_x), np.array(convergence_y),np.array(convergence_z)


# In[6]:


def Histogram_MF(M,bin_values):
   """
   This function performs histogram of given array with respect the bins.
   """
   hist,bin_edges =  np.histogram(M,bins=bin_values,density=False)
   return hist, bin_edges

def bin_size(bins,x,y):  
   """
   This function calculate bin size to weight the quantity.    
   """
   b =[]
   for i, j in zip(x,y):
       b_s = (bins[j]-bins[i])
       b.append(b_s)
   B_s = np.array(b)
   return B_s

def MF_plot(mass,bin_i):
   """
   This function gives mass function of galaxies.
   The mass of galaxies were calculated from new centers calculated from AHF.
   """
   hist, H_bins = Histogram_MF(mass,bin_i)
   xi = np.arange(0,15)
   yi = np.arange(1,16)
   bin_s = (bin_size(np.log(H_bins),xi,yi)) 
   return hist,H_bins,bin_s
   


# In[7]:


def Galaxy_mass_fuction_z5(mass,mass_pivot,phi_str,alpha):
    """
    This function gives stellar mass function for redshift 5.
    Based on Caputi et al 2011.
    """
    phi = []
    for i in mass:
        p = (phi_str * ((i/mass_pivot)**(1-alpha)) * np.exp(-i/mass_pivot))
        phi.append(p)
    PHI = np.array(phi)
    return PHI

def Galaxy_mass_function_z6(mass,mass_pivot,phi_str,alpha):
    """
    Stellar Mass function for redshift 6 and above(till 11).
    Based on Stefanon et al 2021.
    """
    phi = []
    for i in mass:
        p = np.log(10) * phi_str * 10**((i-mass_pivot)*(1+alpha)) * np.exp(-10**(i-mass_pivot))
        phi.append(p)
    PHI = np.array(phi)
    return PHI


# In[8]:


def bbox_generator(coord,radius):
    """
    Creates a box of region of RAMSES data within given boundaries with respect to the center.
    """
    x,y,z = coord
    bb = [[x-radius,y-radius,z-radius],[x+radius,y+radius,z+radius]]
    return bb


# In[9]:


def Gas_profile_data(gas_data,x,y,z,radius):
    """
    Gas_profile_data function returns coordinates of all gas cells, density,temperature,mass and metallicity
    with sphere of radius (0.1*virial radius = Galaxy radius). 
    """
    x_c = x-gas_data[:,0]
    y_c = y-gas_data[:,1]
    z_c = z-gas_data[:,2]
    
    g_x = gas_data[:,0]
    g_y = gas_data[:,1]
    g_z = gas_data[:,2]
    g_rho = gas_data[:,3]
    g_temp = gas_data[:,4]
    g_mass = gas_data[:,5]
    g_metal = gas_data[:,6]
    g_pres = gas_data[:,7]
    
    pos = np.c_[x_c,y_c,z_c] 
    dist = np.linalg.norm(pos,axis=1)
    
    Gas_in_sphere = np.where(dist<=radius)
    
    grad=dist[Gas_in_sphere]
    gDi=g_rho[Gas_in_sphere]
    gTi=g_temp[Gas_in_sphere]
    gMai=g_mass[Gas_in_sphere]
    gmeti=g_metal[Gas_in_sphere]
    gpresi=g_pres[Gas_in_sphere]
    
    gas_profile_D = np.c_[grad*17031.33744,gDi,gTi,gMai,gmeti/0.02,gpresi]
    
    return gas_profile_D


# In[10]:


def temp_profile(gas_p, r_0, r_1):
    """
    temp_profile returns array of temperature profile of galaxy with respect to radial bins.
    The radial bins are in log scale
    """
    temp_p = []
    for i, j in zip(r_0, r_1):
        gtemp = []
        gmass=[]
        for k in gas_p:
            if i < k[0] <= j:
                gtemp.append(k[2])
                gmass.append(k[3])
        
        Gas_temp = np.sum(np.array(gtemp)*np.array(gmass))/np.sum(gmass)
        temp_p.append(Gas_temp)
    return np.array(temp_p)

def pres_profile(gas_p, r_0, r_1):
    """
    pres_profile returns array of pressure profile of galaxy with respect to radial bins.
    The radial bins are in log scale
    """
    pres_p = []
    for i, j in zip(r_0, r_1):
        gpres = []
        gmass = []
        for k in gas_p:
            if i < k[0] <= j:
                gpres.append(k[5])
                gmass.append(k[3])
        Gas_pres = np.sum(np.array(gmass)*np.array(gpres))/np.sum(gmass)
        pres_p.append(Gas_pres)
    return np.array(pres_p)

def dens_profile(gas_p, r_0, r_1):
    """
    dens_profile returns array of density profile of galaxy with respect to radial bins.
    The radial bins are in log scale
    """
    dens_p = []
    Volume = []
    mass_in_shell = []
    for i, j in zip(r_0, r_1):
        gmass = []
        for k in gas_p:
            if i < k[0] < j:
                gmass.append(k[3])
        
        
        Gas_tmass = np.sum(gmass)
        vol = (4/3*np.pi*j**3) - (4/3*np.pi*i**3)
        gdens = Gas_tmass / vol
        dens_p.append(gdens)
        Volume.append(vol)
        mass_in_shell.append(Gas_tmass)
    
    return np.array(dens_p), np.array(Volume),np.array(mass_in_shell)

def metal_profile(gas_p, r_0, r_1):
    """
    metal_profile returns array of metallicity profile of galaxy with respect to radial bins.
    The radial bins are in log scale
    """
    metal_p = []
    for i, j in zip(r_0, r_1):
        gmetal = []
        gmass=[]
        for k in gas_p:
            if i < k[0] <= j:
                gmetal.append(k[4])
                gmass.append(k[3])
        Gas_metal = np.mass(np.array(gmetal)*np.array(gmass))/np.sum(gmass)
        metal_p.append(Gas_metal)
    return np.array(metal_p)


# In[11]:


def star_profile_data(star_data,x,y,z,radius):
    x_c = x-star_data[:,0]
    y_c = y-star_data[:,1]
    z_c = z-star_data[:,2]
    
    s_x = star_data[:,0]
    s_y = star_data[:,1]
    s_z = star_data[:,2]
    s_mass = star_data[:,3]
    s_metal = star_data[:,4]
    
    pos = np.c_[x_c,y_c,z_c] 
    dist = np.linalg.norm(pos,axis=1)
    
    star_in_sphere = np.where(dist<=radius)
    
    srad=dist[star_in_sphere]
    sMai=s_mass[star_in_sphere]
    smeti=s_metal[star_in_sphere]
    
    star_profile_D = np.c_[srad*17031.33744,sMai,smeti/0.02]
    
    return star_profile_D


# In[ ]:


def dens_profile_str(str_p, r_0, r_1):
    dens_ps = []
    Volume = []
    
    for i, j in zip(r_0, r_1):
        str_mass = []
        for k in str_p:
            if i < k[0] < j:
                str_mass.append(k[1])
        
        star_tmass = np.sum(str_mass)
        vol = (4/3*np.pi*j**3) - (4/3*np.pi*i**3)
        str_dens = star_tmass / vol
        dens_ps.append(str_dens)
        Volume.append(vol)
    
    return np.array(dens_ps), np.array(Volume)


def metal_profile_str(str_p, r_0, r_1):
    metal_ps = []
    for i, j in zip(r_0, r_1):
        str_metal = [k[2] for k in str_p if i < k[0] <= j]
        Star_metal = np.median(str_metal)
        metal_ps.append(Star_metal)
    return np.array(metal_ps)


# In[1]:


def DM_profile_data(DM_data,x,y,z,radius):
    x_c = x-DM_data[:,0]
    y_c = y-DM_data[:,1]
    z_c = z-DM_data[:,2]
    
    DM_x = DM_data[:,0]
    DM_y = DM_data[:,1]
    DM_z = DM_data[:,2]
    DM_mass = DM_data[:,3]
    
    
    pos = np.c_[x_c,y_c,z_c] 
    dist = np.linalg.norm(pos,axis=1)
    
    DM_in_sphere = np.where(dist<=radius)
    
    DMrad=dist[DM_in_sphere]
    DMMai=DM_mass[DM_in_sphere]
        
    DM_profile_D = np.c_[DMrad*25279.93059,DMMai]
    
    return DM_profile_D


# In[ ]:


def dens_profile_DM(DM_p, r_0, r_1):
    dens_pdm = []
    Volume = []
    
    for i, j in zip(r_0, r_1):
        DM_mass = 0
        for k in DM_p:
            if i <= k[0] < j:
                DM_mass += k[1]
        
        vol = (4/3) * np.pi * (j**3 - i**3)
        DM_dens = DM_mass / vol
        dens_pdm.append(DM_dens)
        Volume.append(vol)
    
    return np.array(dens_pdm), np.array(Volume)

def dens_profile_DM_with_counts(DM_p, r_0, r_1):
    dens_pdm = []
    P_error = []
    counts = []
    error = []
    
    DM_p = np.array(DM_p)
    r = DM_p[:, 0]
    mass = DM_p[:, 1]
    
    for i, j in zip(r_0, r_1):
        mask = np.where((i < r) & (r < j))
        dm_within_bin = DM_p[mask]
        
        DM_mass = np.sum(dm_within_bin[:, 1])
        num_particles = len(dm_within_bin)
        
        vol = (4/3) * np.pi * (j**3 - i**3)
        DM_dens = DM_mass / vol
        poisson_error = DM_dens*(np.sqrt(num_particles)/num_particles)
        dens_pdm.append(DM_dens)
        P_error.append(poisson_error)
        
    
    return np.array(dens_pdm),np.array(P_error)

def star_SFR_data(star_data,x,y,z,radius):
    x_c = x-star_data[:,0]
    y_c = y-star_data[:,1]
    z_c = z-star_data[:,2]
    
    s_x = star_data[:,0]
    s_y = star_data[:,1]
    s_z = star_data[:,2]
    s_mass = star_data[:,3]
    s_age = star_data[:,4]
    
    pos = np.c_[x_c,y_c,z_c] 
    dist = np.linalg.norm(pos,axis=1)
    
    star_in_sphere = np.where((dist<=radius))
    sMai=s_mass[star_in_sphere]
    sAgi=s_age[star_in_sphere]
    star_SFR_D = np.c_[sMai,sAgi]
    
    return star_SFR_D

def Y_star_SFR_data(star_data,x,y,z,radius,redshift):
    x_c = x-star_data[:,0]
    y_c = y-star_data[:,1]
    z_c = z-star_data[:,2]
    
    s_x = star_data[:,0]
    s_y = star_data[:,1]
    s_z = star_data[:,2]
    s_mass = star_data[:,3]
    s_age = star_data[:,4]
    
    age_Uni = co.t_from_z(redshift).in_units("Myr")
    
    pos = np.c_[x_c,y_c,z_c] 
    dist = np.linalg.norm(pos,axis=1)
    
    star_in_sphere = np.where((dist<=radius) & ((age_Uni.value-s_age)<=20) )
    sMai=s_mass[star_in_sphere]
    sAgi=s_age[star_in_sphere]
    star_SFR_D = np.c_[sMai,sAgi]
    
    return star_SFR_D

def SFR_galaxy(SFR_data,T_bins):
    gal_sfr,bins_t=np.histogram(SFR_data[:,1],T_bins)
    inds=np.digitize(SFR_data[:,1],T_bins)
     
    time_b=(bins_t[:-1,]+bins_t[1:,])/2
    
    sfr = np.array([SFR_data[:,0][inds==j+1].sum()/20e6 for j in range(len(time_b))])
    return sfr,time_b
    