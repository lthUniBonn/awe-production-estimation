# TODO Docstring
#TODO remove old

# Clustering chain
# File name definitions
from config_clustering import file_name_profiles, file_name_freq_distr, cut_wind_speeds_file, cluster_config, data_info
# old frequency file: wind_speed_probability_file = "wind_resource/freq_distribution_v3{}.pickle"

# power curves:
#power_curve_output_file_name = 'output/power_curve_{{i_profile}}_{cluster_config}{data_info}.{{suffix}}'.format(
#    cluster_config=cluster_config, data_info=data_info) # suffix= csv / pickle
power_curve_output_file_name = 'output/power_curve_{{i_profile}}_{cluster_config}{data_info}_successful.{{suffix}}'.format(
    cluster_config=cluster_config, data_info=data_info) # suffix= csv / pickle
#old: power_curve_file = 'output/power_curve{}{}.csv'.format(suffix,i_profile)

# old (for each cluster id i_profile): 'wind_resource/'+'profile{}{}.csv'.format(suffix, i_profile), sep=";"

refined_cut_wind_speeds_file = cut_wind_speeds_file.replace('estimate', 'refined')
#  cut-in/out wind speeds file:
#old: 'output/wind_limits_estimate{}.csv'.format(suffix)
#old: 'output/wind_limits_refined{}.csv'.format(suffix)


