# BHSIM_TOOLKIT

The BHSIM_TOOLKIT is python codes to analayse and plot the cosmological simulation data with supermassive black hole seedings. Currently it's only applicable for Ramses simulation data with limited analyses.

GALCEN_FINDER: This class is used to find the center of all galaxies (currently the halos with more than 100 stellar particles). 
	
	example: galaxy_finder = GalaxyCenterFinder()
                 galaxy_finder.find_galaxy_center(Ramses_simulation_path, AHF_halocatalogue_path, Output_name.extension)
                 
The output file extension could be either txt or csv.

                 
