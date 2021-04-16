from script.tools import *
import json





# Open configurations
with open('experiments/BWfus004_arena.json') as json_file:
    data = json.load(arena_data)

with open('experiments/BWfus004_ezm.json') as json_file:
    data = json.load(ezm_data)


#load raw data
print('Load raw data..')
arena_data = load_all_data(arena_data)
ezm_data   = load_all_data(ezm_data)

#Process Spike Detection
print('Process Spike Detection..')
ezm_data   = process_get_spike_train(ezm_data)
arena_data = process_get_spike_train(arena_data)

#Process EMZ Data
print('Process Datasets..')
ezm_data = process_downsample_by_50(ezm_data)
ezm_data = process_band_pass_filer(ezm_data)
ezm_dat  = process_hilbert_tranform(ezm_data)

#Process ARENA Data
arena_data = process_downsample_by_50(arena_data)
arena_data = process_band_pass_filer(arena_data)
arena_data = process_hilbert_tranform(arena_data)

#align loc and lfp
print('Align Timings...')
ezm_data = process_align_lfp_vs_loc(ezm_data)

#compare the power when
plot_compare(ezm_data, ezm_data)


# Compare power vs location (hue: frq band)
plot_compare_lfp_power_vs_loc(ezm_data,
                            brain_area_1 = 'mPFC',
                            brain_area_2 = 'hipp')
stat_compare_lfp_power_vs_loc(ezm_data,
                            brain_area_1 = 'mPFC',
                            brain_area_2 = 'hipp')

# Compare power coh (hue: frq band)
plot_compare_lfp_power_coh_vs_loc(ezm_data,
                            brain_area_1 = 'mPFC',
                            brain_area_2 = 'hipp')

stat_compare_lfp_power_coh_vs_loc(ezm_data,
                              brain_area_1 = 'mPFC',
                              brain_area_2 = 'hipp')

# Compare phase coh (hue: frq band)
plot_compare_lfp_phase_coh_vs_loc(ezm_data,
                            brain_area_1 = 'mPFC',
                            brain_area_2 = 'hipp')

stat_compare_lfp_phase_coh_vs_loc(ezm_data,
                              brain_area_1 = 'mPFC',
                              brain_area_2 = 'hipp')

# Compare spike vs Loc
plot_compare_fire_rate_vs_loc(ezm_data,
                              brain_area_1 = 'mPFC',
                              brain_area_2 = 'hipp')

#Plots
plot_event   (ezm_data)
plot_spectrum(ezm_data, channel_list = [0,1])
plot_location(ezm_data)
plot_compare_lfp_power_vs_loc(ezm_data)
plot_compare(ezm_data, ezm_data)
plot_travel_distance_over_time([ezm_data, arena_data])
plot_location(arena_data)
plot_compare_lfp_power_coh_vs_loc(arena_data,
                                  brain_area_1 = 'mPFC',
                                  brain_area_2 = 'hipp')

