import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import code
import time

def basic_info(df,dp,dl):
    a = df.meterID.unique()
    b = dl["Meter number"].unique()
    print("=====================================================================================")
    print("Data time range: {} to {}".format(df.recordtime.min(), df.recordtime.max()))
    print("Total no. of meters: {}".format(df.meterID.nunique()))
    print("No. of smart meters in 'All data': {}".format(df[(df.mtype=="Smart Meter")].meterID.nunique()))
    print("No. of grid meters in 'All data': {}".format(df[(df.mtype=="Grid Meter")].meterID.nunique()))
    print("'All data' Grid meter IDs: {}".format(df[(df.mtype=="Grid Meter")].meterID.unique()))
    print("Grid meter in README but not in 'All data': 541810")
    print()
    print("=====================================================================================")
    print("Length of original dataset: {}".format(len(df)))
    print("Length of dataset after preprocessing for duplicates: {}".format(len(dp)))
    print()
    print("=====================================================================================")
    print("Meters present in 'All data' but not in 'Smartmeter list': {}".format(set(a)-set(b)))
    print("Meters present in 'Smartmeter list' but not in 'All data': {}".format(set(b)-set(a)))
    print()
    print("=====================================================================================")
    print("Meters deployed in {} markets: {}".format(dl.Market.nunique(),dl.Market.unique()))
    print("{} different kinds of businesses covered: {}".format(dl["Type of business"].nunique(),dl["Type of business"].unique()))
    print()
    print("=====================================================================================")
    return None

def meter_data_preprocess(df):
    # Converts time values to timestamps and classifies a meter as smart or grid meter.
    print("Preprocessing Started")
    df.recordtime = pd.to_datetime(df.recordtime,format='%d.%m.%Y %H:%M')
    grid_meters = [541807, 541808, 541809, 541810, 541852, 541854, 541870, 541904, 541933, 541938, 541946, 541815]
    df['mtype'] = df.meterID.apply(lambda x: "Grid Meter" if np.isin(x, grid_meters) else "Smart Meter")
    print("Preprocessing Done")
    return df

def event_grouping(df, filtering=False):
    # Filtering lets you choose if you want to analyze the durations based on non-zero consumption only
    if filtering==True:
        df = df[(df.a_current>0.0)]
    ############################################################################
    # all the events whose consecutive timestamps are within 1 hour margin
    # are labeled to be as belonging to the same group
    #
    # ASSUMPTION: If consecutive timestamps are within 1 hour margin then the
    # generator is still on and the missing timestamps during that hour were due to
    # communication failure
    ############################################################################
    df = df.sort_values(by=['mtype', 'meterID','recordtime'])
    # df['new_group'] = df.groupby(['mtype','meterID']).recordtime.apply(lambda x: (x-x.shift(1)).dt.total_seconds()>3600.0).cumsum()
    df['new_group'] = (((df.recordtime-df.recordtime.shift(1)).dt.total_seconds()>3600.0) | (df.meterID != df.meterID.shift(1))).cumsum()
    # grouping by the new groups formed to calculate duration of typical generator usage
    dg = df.groupby(['mtype','meterID','new_group']).agg({'recordtime':['min','max','count'], 'meterCount':['min','max'], 'a_current':['min','mean','max'], 'a_active_power':['min','mean','max'], 'a_power_factor':['min','mean','max'], 'a_voltage':['min','mean','max']}).reset_index()
    dg.columns = ['mtype','meterID','new_group','t_min','t_max','packs_tx','kwh_min', 'kwh_max', 'i_min', 'i_mean', 'i_max', 'p_min', 'p_mean', 'p_max', 'pf_min', 'pf_mean', 'pf_max', 'v_min', 'v_mean', 'v_max']
    dg['dur'] = (dg['t_max'] - dg['t_min']).dt.total_seconds()/3600.00 # duration in hours
    # one packet expected at every one minute timestamp
    dg['packs_exp'] = dg['dur'].apply(lambda x: (x*60 + 1)) # +1 to add the last timestamp of the event
    return dg

def reception_rate_analysis(dg):
    ################Event-level reception rate#####################################
    # we take a mean of event-wise reception rates per meter
    de = dg.copy()
    de['recep_eve'] = de.packs_tx*100.0/de.packs_exp
    de = de.groupby(['mtype','meterID']).mean()[['recep_eve']].reset_index()
    ################Overall reception rate#########################################
    # For every meter, we divide the total samples received to total expected samples
    # to obtain overall meter reception rate
    do = dg.groupby(['mtype','meterID']).sum()[['packs_tx','packs_exp']].reset_index()
    do['recep_ov'] = do.packs_tx*100.0/do.packs_exp
    ###############DISTRIBUTION PLOTS#############################################
    cdf_plot(do,'recep_ov','Packet Reception Rate (%)','Proportion of Meters (%)',"CDF of Packet Reception Rate")
    # code.interact(local =locals())
    return None


def cdf_plot(df,column,xlabel,ylabel,title):
    df = df.sort_values(by=column)
    df = df.reset_index()
    df = df.drop(columns=['index'])
    df['ct'] = df.index
    df.ct = df.ct*100.0/df.ct.max()
    plt.plot(df[column], df.ct)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    code.interact(local = locals())
    return None


def event_duration(dg,sampling_f):
    # Returns average duration of generator operation for the given sampling frequency - daily, monthly
    dg = dg[(dg.mtype == "Smart Meter")] #filtering out grid meters
    dd = dg.groupby(['mtype','meterID']).resample(sampling_f, on='t_min').sum()[['dur']].reset_index()
    dd.columns = ['mtype', 'meterID', 't_min', 'total_dur']
    dd = dd.groupby(['mtype', 'meterID']).mean()[['total_dur']].reset_index()
    return dd

def filt_unfilt_duration(dg1, dg2, freq):
    df1 = event_duration(dg1, freq)#without filtering
    df2 = event_duration(dg2, freq)#with filtering
    df2 = df2.rename(columns = {'total_dur':'dur_supply'})
    dh = pd.merge(df1,df2,left_on='meterID',right_on='meterID',how='outer')
    dh = dh.fillna(0)
    dh['dur_zero_use'] = dh.total_dur - dh.dur_supply
    dh = dh.sort_values(by='total_dur')
    dh = dh[['meterID','dur_supply','dur_zero_use']]
    dh.index = dh.meterID
    dh.loc[:,['dur_supply','dur_zero_use']].plot.bar(stacked=True, figsize=(16,7))
    plt.legend(["Non-zero consumption recorded","No consumption recorded"])
    plt.ylabel("Duration (hours)")
    plt.xlabel("Meter ID")
    plt.title("Average monthly duration for which a generator was ON")
    plt.show()
    code.interact(local = locals())
    return None

def duration_cdf(df):
    df = event_duration(df, '1M')
    df = df.sort_values(by='total_dur')
    # code.interact(local = locals())
    df = df.reset_index()
    df = df.drop(columns = ['index'])
    df['ct'] = df.index
    df.ct = df.ct*100.0/df.ct.max()
    plt.plot(df.total_dur, df.ct)
    plt.xlabel('Average Monthly Duration of Generator Operation')
    plt.ylabel('% of Generators')
    plt.grid()
    plt.show()
    # plt.savefig('generators_cdf.pdf')
    return None

def typical_lifetime(df, title):
    df = df[(df.mtype == "Smart Meter")] #filtering out grid meters
    df = df.sort_values(by=['meterID'])
    df.boxplot(column = ['dur'], by=['meterID'])
    plt.grid(False)
    plt.xlabel('Meter ID')
    plt.ylabel('Duration (hours)')
    plt.title(title)
    plt.suptitle("")
    plt.show()
    return None

def power_util(df,title):
    df = df[(df.mtype=="Smart Meter")]
    df = df.sort_values(by=['meterID'])
    df.boxplot(column = ['a_active_power'], by=['meterID'])
    plt.grid(False)
    plt.xlabel('Meter ID')
    plt.ylabel('Power Consumption (kW)')
    plt.title(title)
    plt.suptitle("")
    plt.show()
    return None

def time_of_use(df):
    df = df.copy()
    df['tou'] = df.recordtime.apply(lambda x: "Morning" if 6<=x.hour<12 else "Noon" if 12<=x.hour<17 else "Evening" if 17<=x.hour<23 else "Night")
    df = df.groupby(['meterID','tou']).count()[['meterCount']].reset_index()
    code.interact(local = locals())
    return None

if __name__ == '__main__':
    CURRENTDIR = os.getcwd()
    re = CURRENTDIR + '/data/release_1/'
    #-----Read & Process Smart Meter Data----------
    '''df = pd.read_csv(re + '003_Release_1_all_data.csv')
    db = meter_data_preprocess(df.copy())
    outfile = open(re + 'smartmeter_data', 'wb')
    pickle.dump(db, outfile)
    outfile.close()'''
    db = pd.read_pickle(re+'smartmeter_data')
    # further preprocessing to avoid timestamp duplicates per meter
    dp = db.groupby(['mtype','meterID','recordtime']).mean().reset_index()
    dp = dp.drop(columns = ['index'])
    #-----Read List Information--------------------
    dlist = pd.read_excel(re + '002_ Release_1_Smartmeter_List.xlsx')
    #-----Basic Information Printing-------------------------
    basic_info(db,dp,dlist)
    #------Forming Event Groups------------------------------
    dg = event_grouping(dp.copy()) #Contains duration for which generator was ON
    # If mean current for the group is 0, it means no supply was used during that interval
    dg_f = dg[(dg.i_mean>0.0)] #Filters out groups for which generator was ON and there was some current flowing atleast for sometime
    #------duration & frequency analysis---------------------
    # filt_unfilt_duration(dg, dg_f,'1M') #1 day frequency doesn't make sense for now for this plot
    # non_zero_consumption_duration(dn.copy(), '1M')
    # plot cdf of number of generators by monthly duration
    # duration_cdf(dg.copy())
    #-------Typical Lifetime of operation--------------------
    # typical_lifetime(dg.copy(), "Distribution of a Generator's Typical Operation Duration \n (Operation means generator in ON state)")
    # typical_lifetime(dg_f.copy(), "Distribution of a Generator's Typical Operation Duration \n (Filter: generator in ON state and consumption is non-zero)")
    #-------Time of Use--------------------------------------
    # time_of_use(dg1.copy())
    #-------Power Recorded-----------------------------------
    # power_util(dp.copy(),"Distribution of active power demand per generator")
    #-----Time of Use----------------------------------------
    # time_of_use(dp.copy())
    #---------Reception Rate---------------------------------
    reception_rate_analysis(dg)
    code.interact(local = locals())
