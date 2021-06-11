# TODO Docstring
#TODO remove old

# Clustering chain

# File name definitions
from config_clustering import file_name_profiles, file_name_freq_distr, cut_wind_speeds_file, data_info, result_dir, plots_interactive, \
    n_clusters

#optimizer:
optimizer_history_file_name = result_dir + 'optimizer_history_{data_info}.hist'.format(data_info=data_info)

# power curves:
run_marker = '_opt-successful-test' # default = ''
power_curve_output_file_name = result_dir + 'power_curve_{{i_profile}}_{data_info}{run_info}_filtered.{{suffix}}'.format(
    data_info=data_info, run_info=run_marker) # suffix= csv / pickle

refined_cut_wind_speeds_file = cut_wind_speeds_file.replace('estimate', 'refined')

plot_output_file = result_dir + '{title}' + data_info + '.pdf'


