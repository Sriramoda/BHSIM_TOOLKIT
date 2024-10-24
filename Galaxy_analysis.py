#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries
import numpy as np


# In[5]:


# Functions used to extract the properties of galaxy from halo finder outputs
## Currently the code is wrote only for AHF output file - .AHF_halos file
"""
Criteria for Halo selection are:
1. Only Host halos(no substructures)
2. Only halos that have non zero stellar mass
3. Only halos that have more than 100 star particles
"""

def halo_center(data):
    # halo_center function fetches x,y,z coordinates of halo center from AHF catalog.
    
    x_c=[]
    y_c=[]
    z_c=[]
    all_halo_center = np.c_[data[:,5],data[:,6],data[:,7],data[:,0]]
    for halos in data:
        if halos[1]==-1:
            if halos[64]>0:
                if halos[63]>100:
                    x_c.append(halos[5])
                    y_c.append(halos[6])
                    z_c.append(halos[7])
    Halo_center = np.vstack(np.meshgrid((x_c,y_c,z_c))).reshape(3,-1).T
    return Halo_center

def halo_virial_radius(data):

    # halo_virial_radius function fetches virial radius from AHF catalog for desired halos.

    rvir=[]
    for halos in data:
        if halos[1]==-1:
            if halos[64]>0:
                if halos[63]>100:
                    rvir.append(halos[11])
    R_vir = np.array(rvir)
    return R_vir

def halo_id(data):

    # halo_id function fetches ID from AHF catalog for desired halos.

    id_=[]
    for halos in data:
        if halos[1]==-1:          
            if halos[64]>0:
                if halos[63]>100:
                    id_.append(halos[0])
    ID = np.array(id_)
    return ID

def halo_mass(data):
    
    #halo_id function fetches ID from AHF catalog for desired halos.
    
    H_mass=[]
    for halos in data:
        if halos[1]==-1:          
            if halos[64]>0:
                if halos[63]>100:
                    H_mass.append(halos[3])
    Halo_mass = np.array(H_mass)
    return Halo_mass


def convergence_test(xo,yo,zo,xn,yn,zn):
    """
    convergence_test finds convergence between old and new center of mass from galaxy_com_center.
    This function works within galaxy com center
    """
    x_conv = np.abs(xo-xn)
    y_conv = np.abs(yo-yn)
    z_conv = np.abs(zo-zn)
    return x_conv,y_conv,z_conv

def galaxy_COM(center, factor, Rvir,id_,H_mass,strm):
    """
    galaxy_com_center function finds center of galaxy located within halo by calculating COM of
    all stellar particles that lies within 0.1 times virial radius of halo.
    
    center = Halo centers from AHF with units (kpc/h)
    
    factor = 0.1 or less, (0.1*Rvir) approximately equal to galactic radius
    
    Rvir =  Virial radius of halo from AHF with units(kpc/h)

    H_mass = Total mass of halo (M_sol/h)
    
    strm = array of stellar particle coordinates and mass with shape of [n,4] and units [kpc/h,kpc/h,kpc/h,Msun]
    """
    x_c,y_c,z_c = [],[],[]
    rv   = []
    idd  = [] 
    h_m = []
    MGal_tot = []
    convergence_x=[]
    convergence_y=[]
    convergence_z=[]
    
    for c,r,i,m in zip(center,Rvir,id_,H_mass):
        stellar_rel_xdist = c[0] - strm[:,0]
        stellar_rel_ydist = c[1] - strm[:,1]
        stellar_rel_zdist = c[2] - strm[:,2]
        mass_star = strm[:,3]
        star_x = strm[:,0]
        star_y = strm[:,1]
        star_z = strm[:,2]
    
        pos = np.c_[stellar_rel_xdist, stellar_rel_ydist, stellar_rel_zdist] 
    
        dist = np.linalg.norm(pos,axis=1)   # Distance of each stellar particle from the center of halo from AHF
        Convergence_radius = factor*r       # Factor is reduced for each repetition to converge the center of galaxy
        mass_in_Rgal = 0.1*r                # criteria for calculating mass of particles within the given radius of galaxy (10% of virial radius)
    
        Star_inside = np.where(dist<=Convergence_radius)
        mass_inside = np.where(dist<=mass_in_Rgal)
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
            MGal_tot.append(np.sum(full_mass))
            rv.append(r)
            idd.append(i)
            h_m.append(m)
            conv_x,conv_y,conv_z = convergence_test(c[0],c[1],c[2],x_new,y_new,z_new)
            convergence_x.append(conv_x)
            convergence_y.append(conv_y)
            convergence_z.append(conv_z)
            
        
    
    return np.array(x_c), np.array(y_c), np.array(z_c),np.array(rv),np.array(idd),np.array(MGal_tot),np.array(h_m),np.array(convergence_x), np.array(convergence_y),np.array(convergence_z)


# In[ ]:




