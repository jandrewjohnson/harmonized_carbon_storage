import os
import numpy as np
import hazelbean as hb

base_data_folder = 'c:\\OneDrive\\Projects\\base_data'
ipcc_uri = os.path.join(base_data_folder, r"carbon\ruesch_and_gibbs\ruesch_and_gibbs_mg_c_per_ha_global_30s.tif")

avitabile_uri = os.path.join(base_data_folder, 'carbon\\avitabile\\Avitabile_AGB_Map.tif')
geocarbon_uri = os.path.join(base_data_folder, 'carbon\\avitabile\\GEOCARBON_Global_Forest_Biomass\\GEOCARBON_Global_Forest_AGB_10072015.tif')

# Set folders
temp_folder = 'C:\\temp'
run_folder = os.path.join(temp_folder, 'run_' + hb.random_string())
os.mkdir(run_folder)
intermediate_folder = os.path.join(base_data_folder, 'carbon\\johnson\\decision_tree_combined_carbon')

# Open fixed inputs as arrayframes
ipcc = hb.ArrayFrame(ipcc_uri)
avitabile = hb.ArrayFrame(avitabile_uri)
geocarbon = hb.ArrayFrame(geocarbon_uri)

# Additional resources to calculate totals
ha_per_cell_30s_uri = os.path.join(base_data_folder, 'misc\\ha_per_cell_30s.tif')
land_ha_per_cell_30s_uri = os.path.join(base_data_folder, 'misc\\land_ha_per_cell_30s.tif')
ha_per_cell = hb.ArrayFrame(ha_per_cell_30s_uri)

# Logic on abg, c conversions.
carbon_abg_proportion_common_value = 0.5 # What saatchi used.

explanation_for_carbon_abg_proportion = """From Djomo et al: The forest carbon stocks are 
widely estimated from the allometric
equations for forest biomass. Generally, the carbon concentration of the
different parts of a tree is assumed to be 50% of the biomass [42] or 45%
of the biomass [43]. However, Losi et al. [44] in their study estimated
the carbon concentration of dry bole sample to be approximately 48%
of the dry bole biomass. Djomo et al. [38] analyses the carbon content
in wood with a CNS analyser and found a mean value of 46.53%. The
biomass estimation of the forest can be worked out using any of the
methods or in combination of the methods mentioned. At the same
time, while choosing a method for biomass estimation one should keep
in mind the applicability or the suitability of that method for the area
or forest type or tree species. The allometric equations and regression
models, for biomass estimation, also should not be used beyond their
range of validity [22,45]. """
carbon_abg_proportion = .4653 # Based on Dejovo and Gravenhorst 2011, summarized by Vashum 2012 - Using Methods to Estimate Above-Ground Biomass and Carbon Stock in Natural Forests - A Review



# Resample to match the IPCC extent and resolution
make_avitabile_compatible_to_ipcc = 1
if make_avitabile_compatible_to_ipcc:
    print('       Resampling Avitable.')
    avitabile_compatible_uri = os.path.join(run_folder, 'avitabile_compatible.tif')

    hb.resample_to_match(avitabile.path,
                         ipcc.path,
                         avitabile_compatible_uri,
                         resample_method='near',
                         output_data_type=6,
                         src_ndv=None,
                         ndv=-9999.0,
                         compress=True,
                         ensure_fits=False,
                         gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS,
                         calc_raster_stats=False,
                         add_overviews=False,
                         pixel_size_override=None,
                         verbose=False,)
    avitabile_compatible = hb.ArrayFrame(avitabile_compatible_uri)
else:
    print('       Loading existing Avitable.')
    avitabile_compatible_uri = os.path.join(intermediate_folder, 'avitabile_compatible.tif')
    avitabile_compatible = hb.ArrayFrame(avitabile_compatible_uri)

# Convert from Above-Ground Biomass (AGB) to carbon content
scale_avitabile = 1
if scale_avitabile:
    avitabile_scaled_uri = os.path.join(run_folder, 'avitabile_mg_c_per_ha_global_30s.tif')
    avitabile_scaled = hb.multiply(avitabile_compatible.path, carbon_abg_proportion, avitabile_scaled_uri)
else:
    print('       Loading existing Avitable.')
    avitabile_scaled_uri = os.path.join(intermediate_folder, 'avitabile_mg_c_per_ha_global_30s.tif')
    avitabile_compatible = hb.ArrayFrame(avitabile_scaled_uri)

# Repeat resample and rescaling for geocarbon
make_geocarbon_compatible_to_ipcc = 1
if make_geocarbon_compatible_to_ipcc:
    print('       Resampling geocarbon.')
    geocarbon_compatible_uri = os.path.join(run_folder, 'geocarbon_compatible.tif')

    hb.resample_to_match(geocarbon.path,
                         ipcc.path,
                         geocarbon_compatible_uri,
                         resample_method='bilinear',
                         output_data_type=6,
                         src_ndv=None,
                         ndv=-9999.0,
                         compress=True,
                         ensure_fits=False,
                         gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS,
                         calc_raster_stats=False,
                         add_overviews=False,
                         pixel_size_override=None,
                         verbose=False, )

    geocarbon_compatible = hb.ArrayFrame(geocarbon_compatible_uri)
else:
    print('       Loading existing geocarbon.')
    geocarbon_compatible_uri = os.path.join(intermediate_folder, 'geocarbon_compatible.tif')
    geocarbon_compatible = hb.ArrayFrame(geocarbon_compatible_uri)

scale_geocarbon = 1
if scale_geocarbon:
    print('       Scaling geocarbon.')
    geocarbon_scaled_uri = os.path.join(run_folder, 'geocarbon_mg_c_per_ha_global_30s.tif')
    geocarbon_scaled = hb.multiply(geocarbon_compatible.path, carbon_abg_proportion, geocarbon_scaled_uri)
else:
    print('       Loading existing geocarbon.')
    geocarbon_scaled_uri = os.path.join(intermediate_folder, 'geocarbon_mg_c_per_ha_global_30s.tif')
    geocarbon_scaled = hb.ArrayFrame(geocarbon_scaled_uri)

# Apply decision tree
combine_inputs = 1
if combine_inputs:
    print('       Applying decision tree to combine inputs to define carbon_per_ha')
    carbon_combined_step_1_uri = os.path.join(run_folder, 'carbon_above_ground_mg_per_ha_global_30s_step_1.tif')
    carbon_combined_uri = os.path.join(run_folder, 'carbon_above_ground_mg_per_ha_global_30s.tif')

    ipcc_scaled = hb.ArrayFrame(ipcc_uri)

    # Decision tree logic here
    carbon_per_ha = np.where(avitabile_scaled.data > 0, avitabile_scaled.data, geocarbon_scaled.data)
    carbon_per_ha = np.where(carbon_per_ha > 0, carbon_per_ha, ipcc_scaled.data)
    hb.create_af_from_array(carbon_per_ha, carbon_combined_uri, avitabile_scaled)
    # hb.ArrayFrame(carbon_per_ha, avitabile_scaled, output_uri=carbon_combined_uri)
else:
    print('       Loading existing carbon_per_ha.')
    carbon_combined_uri = os.path.join(intermediate_folder, 'carbon_above_ground_mg_per_ha_global_30s.tif')
    carbon_per_ha = hb.ArrayFrame(carbon_combined_uri)


calculate_carbon_per_cell = 1
if calculate_carbon_per_cell:
    print('       Calculating carbon_per_cell.')
    carbon_per_ha = hb.ArrayFrame(carbon_combined_uri)
    carbon_per_cell = hb.multiply(ha_per_cell_30s_uri, carbon_per_ha.path, os.path.join(run_folder, 'carbon_per_cell.tif'))
else:
    print('       Skipping calculate_per_cell.')
    carbon_per_cell_uri = os.path.join(intermediate_folder, 'carbon_per_cell.tif')
    carbon_per_cell = hb.ArrayFrame(carbon_per_cell_uri)

# carbon_per_cell assumes the carbon is spread evenly through ALL the cell. The following uses NASA's representation of
# land-ha per cell. This calculation has ZERO difference from above, thus I will not include it in any data.
calculate_carbon_per_cell_using_land_ha = False
# if calculate_carbon_per_cell_using_land_ha:
#     print('       Calculating carbon_per_cell.')
#     carbon_per_cell_using_land_ha = hb.multiply(land_ha_per_cell_30s, carbon_per_ha, output_uri=os.path.join(run_folder, 'carbon_per_cell_using_land_ha.tif'))
#     carbon_per_cell_using_land_ha = carbon_per_cell_using_land_ha * 100
# else:
#     print('       Skipping calculate_per_cell.')
#     carbon_per_cell_using_land_ha_uri = os.path.join(intermediate_folder, 'carbon_per_cell_using_land_ha.tif')
#     carbon_per_cell_using_land_ha = hb.ArrayFrame(carbon_per_cell_using_land_ha_uri)

calculate_ipcc_carbon_per_cell = 1
if calculate_ipcc_carbon_per_cell:
    ipcc_per_cell_uri = output_uri=os.path.join(run_folder, 'ipcc_carbon_per_cell.tif')
    ipcc_per_cell = hb.multiply(ipcc_scaled.path, ha_per_cell_30s_uri, ipcc_per_cell_uri)
else:
    ipcc_per_cell_uri = output_uri = os.path.join(intermediate_folder, 'ipcc_carbon_per_cell.tif')
    ipcc_per_cell = hb.ArrayFrame(ipcc_per_cell_uri)


calculate_total_carbon = 1
if calculate_total_carbon:
    print('       Calculating calculate_total_carbon.')
    print('carbon total', carbon_per_cell.sum()) # Gives carbon total 336557408661.0
    print('ipcc_carbon total', ipcc_per_cell.sum()) # Gives ipcc_carbon total 502387302619.0
else:
    print('       Skipping calculate_total_carbon.')





print('Script Finished.')