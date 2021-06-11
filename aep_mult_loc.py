import numpy as np
import pandas as pd
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cbook import MatplotlibDeprecationWarning
from mpl_toolkits.basemap import Basemap

from config import file_name_freq_distr, power_curve_output_file_name, plots_interactive, result_dir, \
    n_clusters, data_info
from config_clustering import  locations

import warnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def get_mask_discontinuities(df):
    """Identify discontinuities in the power curves. The provided approach is obtained by trial and error and should
    be checked carefully when applying to newly generated power curves."""
    mask = np.concatenate(((True,), (np.diff(df['P [W]']) > -5e2)))
    mask = np.logical_or(mask, df['v_100m [m/s]'] > 10)  # only apply mask on low wind speeds
    if df['P [W]'].iloc[-1] < 0 or df['P [W]'].iloc[-1] - df['P [W]'].iloc[-2] > 5e2:
        mask.iloc[-1] = False
    #print('mask: ', mask)
    return ~mask



def plot_aep_matrix(freq, power, aep, plot_info=''):
    """Visualize the annual energy production contributions of each wind speed bin."""
    n_clusters = freq.shape[0]
    mask_array = lambda m: np.ma.masked_where(m == 0., m)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 3.5))
    plt.subplots_adjust(top=0.98, bottom=0.05, left=0.065, right=0.98)
    ax[0].set_ylabel("Cluster label [-]")
    ax[0].set_yticks(range(n_clusters))
    ax[0].set_yticklabels(range(1, n_clusters+1))

    for a in ax:
        a.set_xticks((0, freq.shape[1]-1))
        a.set_xticklabels(('cut-in', 'cut-out'))

    im0 = ax[0].imshow(mask_array(freq), aspect='auto')
    cbar0 = plt.colorbar(im0, orientation="horizontal", ax=ax[0], aspect=12, pad=.17)
    cbar0.set_label("Probability [%]")
    im1 = ax[1].imshow(mask_array(power)*1e-3, aspect='auto')
    cbar1 = plt.colorbar(im1, orientation="horizontal", ax=ax[1], aspect=12, pad=.17)
    cbar1.set_label("Power [kW]")
    im2 = ax[2].imshow(mask_array(aep)*1e-6, aspect='auto')
    cbar2 = plt.colorbar(im2, orientation="horizontal", ax=ax[2], aspect=12, pad=.17)
    cbar2.set_label("AEP contribution [MWh]")
    if not plots_interactive: plt.savefig(result_dir + 'aep_production_conribution' + plot_info + '.pdf')

def evaluate_aep(n_clusters=8):
    """Calculate the annual energy production for the requested cluster wind resource representation. Reads the wind
    speed distribution file, then the csv file of each power curve, post-processes the curve, and numerically integrates
    the product of the power and probability curves to determine the AEP."""

    with open(file_name_freq_distr, 'rb') as f:freq_distr = pickle.load(f)
    freq_full = freq_distr['frequency']
    wind_speed_bin_limits_full = freq_distr['wind_speed_bin_limits']

    loc_aep = []
    for i_loc, loc in enumerate(locations):
        # Select location data
        if len(locations) == 1:
            freq = freq_full
            wind_speed_bin_limits = wind_speed_bin_limits_full
        else:
            freq = freq_full[i_loc, :, :]
            wind_speed_bin_limits = wind_speed_bin_limits_full[i_loc, : , :]

        p_bins = np.zeros(freq.shape)
        for i in range(n_clusters):
            i_profile = i + 1
            # Read power curve file
            df = pd.read_csv(power_curve_output_file_name.format(suffix='csv', i_profile=i_profile), sep=";")
            #mask_faulty_point = get_mask_discontinuities(df)
            v = df['v_100m [m/s]'].values #.values[~mask_faulty_point]
            p = df['P [W]'].values #.values[~mask_faulty_point]
            while True:
                mask_power_disc = [True] + list(np.diff(p) > -500)
                p = p[mask_power_disc]
                v = v[mask_power_disc]
                if sum(mask_power_disc) == len(mask_power_disc):
                    # No more discontinuities
                    break
                #print('Run power masking again') # masking happens fairly often for now
    
            # assert v[0] == wind_speed_bin_limits[i, 0] #TODO differences at 10th decimal threw assertion error
            err_str = "Wind speed range of power curve {} is different than that of probability distribution: " \
                      "{:.2f} and {:.2f} m/s, respectively."
            if np.abs(v[0] - wind_speed_bin_limits[i, 0]) > 1e-6:
                print(err_str.format(i_profile, wind_speed_bin_limits[i, 0], v[0]))        
            if np.abs(v[-1] - wind_speed_bin_limits[i, -1]) > 1e-6:
                print(err_str.format(i_profile, wind_speed_bin_limits[i, -1], v[-1]))
            # assert np.abs(v[-1] - wind_speed_bin_limits[i, -1]) < 1e-6, err_str
    
            # Determine wind speeds at bin centers and corresponding power output.
            v_bins = (wind_speed_bin_limits[i, :-1] + wind_speed_bin_limits[i, 1:])/2.
            p_bins[i, :] = np.interp(v_bins, v, p, left=0., right=0.)
    
        # Weight profile energy production with the frequency of the cluster 
        aep_bins = p_bins * freq/100. * 24*365
        aep_sum = np.sum(aep_bins)*1e-6
        loc_aep.append(aep_sum)
        print("AEP: {:.2f} MWh".format(aep_sum))
        if i_loc % 100 == 0:
            plot_aep_matrix(freq, p_bins, aep_bins, plot_info=(data_info+str(i_loc)))
            print('AEP matrix plotted for location number {} of {} - at lat {}, lon {}'.format(i_loc, len(locations), loc[0], loc[1]))

    return loc_aep


if __name__ == "__main__":
    # TODO include this into aep from own plotting plot_power_and_wind_speed_probability_curves()
    
    loc_aep = evaluate_aep(n_clusters)
    # plot location map aep
    print('Location wise AEP determined. Plot map:')

    # General plot settings.
    map_resolution = 'i' # Options for resolution are c (crude), l (low), i (intermediate), h (high), f (full) or None


    #Europe map 
    map_lons = [-20, 20]
    map_lats = [65, 30]
   
    # Compute relevant map projection coordinates
    lats = [lat for lat,_ in locations]
    lons = [lon for _,lon in locations]

    cm=1/2.54
    fig = plt.figure(figsize=(13*cm, 14.5*cm))
    ax = fig.add_subplot(111)

    plt.title("AEP for {} clusters".format(n_clusters))

    map_plot = Basemap(projection='merc', llcrnrlon=np.min(map_lons), llcrnrlat=np.min(map_lats), urcrnrlon=np.max(map_lons),
                       urcrnrlat=np.max(map_lats), resolution=map_resolution, ax=ax)
    grid_x, grid_y = map_plot(lons, lats)  # Compute map projection coordinates.

    color_map = plt.get_cmap('nipy_spectral') #continuous 

    normalize = mpl.colors.Normalize(vmin=min(loc_aep), vmax=max(loc_aep))

    map_plot.scatter(grid_x, grid_y, c=color_map(normalize(loc_aep)))

    map_plot.drawcoastlines(linewidth=.4)

    cax, _ = mpl.colorbar.make_axes(ax)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=color_map, norm=normalize, label='AEP [MWh]') 
    
    plt.show()
