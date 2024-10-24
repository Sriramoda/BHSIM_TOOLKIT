import numpy as np
import pandas as pd
import yt
import Galaxy_analysis as GA
import os

class GalaxyCenterFinder:
    def __init__(self):
        # Constructor remains empty, as no need for paths during initialization
        pass

    def load_data(self, Ramses_snapshot_path, AHF_halos_path):
        """
        Loads the Ramses simulation snapshot and AHF halo finder data.
        """
        # Loading the Ramses simulation snapshot and AHF halo data
        print(f"Loading Ramses snapshot from: {Ramses_snapshot_path}")
        self.Ramses_snapshot_data = yt.load(Ramses_snapshot_path)
        print(f"Loading AHF halo data from: {AHF_halos_path}")
        self.AHF_halo_data = np.loadtxt(AHF_halos_path, skiprows=1)

        # Extract all data from the Ramses snapshot
        self.Ramses_snapshot_data_all = self.Ramses_snapshot_data.all_data()

    def find_galaxy_center(self, Ramses_snapshot_path, AHF_halos_path, output_file):
        """
        Finds the galaxy center based on the provided Ramses snapshot and AHF halo data.
        Outputs the center of the galaxy into a text or CSV file.
        """
        # Step 1: Load data
        self.load_data(Ramses_snapshot_path, AHF_halos_path)

        # Step 2: Extract halo properties
        print("Extracting halo properties...")
        Halo_center = GA.halo_center(self.AHF_halo_data) / 17031.3376
        Halo_Rvir = GA.halo_virial_radius(self.AHF_halo_data) / 17031.3376
        Halo_id = GA.halo_id(self.AHF_halo_data)
        Halo_mass = GA.halo_mass(self.AHF_halo_data)

        # Step 3: Extract star positions and masses from the Ramses data
        print("Extracting star data...")
        Star_positions = np.array(self.Ramses_snapshot_data_all['star', 'particle_position'])
        Star_mass = np.array(self.Ramses_snapshot_data_all['star', 'particle_mass'].in_units('Msun'))
        Str_mass_pos = np.c_[Star_positions, Star_mass]

        # Step 4: Calculate galaxy center of mass (CoM)
        print("Calculating galaxy center of mass...")
        x1, y1, z1, Rv1, id_1, gal_mass, hm1, x_conver, y_conver, z_conver = GA.galaxy_COM(
            Halo_center, 0.1, Halo_Rvir, Halo_id, Halo_mass, Str_mass_pos
        )

        x2, y2, z2, Rv2, id_2, gal_mass, hm2, x_conver, y_conver, z_conver = GA.galaxy_COM(
            np.c_[x1, y1, z1], 0.05, Halo_Rvir, Halo_id, Halo_mass, Str_mass_pos
        )

          # Step 4: Create data for saving
        Galaxy_data_smbh_array = np.array([id_2, x2, y2, z2, Rv2, gal_mass, hm2]).T  # Transpose to get data in columns
        
        column_names = ['ID', 'x', 'y', 'z', 'Virial radius', 'galaxy mass', 'Halo mass']

        # Step 5: Save data as a CSV or TXT file
        print(f"Saving galaxy center data to: {output_file}")
        if output_file.endswith('.csv'):
            df = pd.DataFrame(Galaxy_data_smbh_array, columns=column_names)
            df.to_csv(output_file, index=False)
        elif output_file.endswith('.txt'):
            np.savetxt(output_file, Galaxy_data_smbh_array, header=' '.join(column_names), fmt='%f')
        else:
            raise ValueError("Unsupported file format. Please use .csv or .txt")

        print(f"Galaxy center data saved successfully to {output_file}.")
        
if __name__ == "__main__":
    # Create an instance of GalaxyCenterFinder
    galaxy_finder = GalaxyCenterFinder()



