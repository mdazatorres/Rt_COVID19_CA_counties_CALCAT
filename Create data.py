import pandas as pd
import numpy as np
import datetime
import pgeocode
save_path = "data/"
read_path = "data/"
donwload = True
if donwload:
    data_all_ww = pd.read_csv( "https://data.ca.gov/dataset/b8c6ee3b-539d-4d62-8fa2-c7cd17c16656/resource/16bb2698-c243-4b66-a6e8-4861ee66f8bf/download/master-covid-public.csv")
    data_all_ww.to_csv(read_path + 'data_cdph.csv')

    data_all_ww_scan = pd.read_csv("http://publichealth.verily.com/api/csv")
    data_all_ww_scan.to_csv(read_path + 'data.csv')
else:
    data_all_ww = pd.read_csv(read_path+"data_cdph.csv")
    data_all_ww_scan = pd.read_csv(read_path + "data.csv")


# Define columns of interest for scan data and CDPH data
cols_scan = ['Date', 'City', 'zipcode', 'Population_Served', 'Plant', 'SC2_N_norm_PMMoV', 'SC2_N_gc_g_dry_weight', 'PMMoV_gc_g_dry_weight', "County_FIPS"]
cols_CDPH = ['Date', 'Plant', 'Population_Served', 'County', 'zipcode', 'pcr_gene_target',  'SC2_N_gc_g_dry_weight', 'PMMoV_gc_g_dry_weight']

#del_counties = ['San Joaquin',  'Ventura', 'Mono', 'Butte', 'Placer', 'Mariposa',  'Plumas', 'Sutter','Shasta','Kern','Imperial']

del_counties = ['San Joaquin', 'Ventura', 'Mono', 'Placer', 'Mariposa',  'Plumas', 'Sutter', 'Shasta',
                'Kern', 'Imperial', 'Del Norte']


# San Joaquin 37 data ( as october 18)
# Ventura does not have PPMoV
# Mono does not have the positivity rate
# we deleded following counties for having  less than 42 data that is the minimum data to use the model
# Butte largest Plant -- 13 data
# Placer 5 data ( as october 18)
# Mariposa -- 13 data  ( as october 18)
#--------------
# Plumas I am going to run again
# Sutter 17 data ( as october 18)
# Shasta 19 data ( as october 18)
# Kern, Imperial no data anymore
# Del Norte


# Rename columns in the scan data to match the CDPH data columns
# For Butte we are going to use 'OrovilleSC' instead of 'Chico_WPCP', that is the largest WWTP
#
data_all_ww_scan = data_all_ww_scan.rename(
                        columns={'Collection_Date': 'Date', 'Zipcode': 'zipcode', 'Plant':'City_loc', 'Site_Name': 'Plant'})

# Rename columns in the CDPH data to match the scan data columns
data_all_ww = data_all_ww.rename(
                columns={'sample_collect_date': 'Date', 'wwtp_name': 'Plant', 'county_names': 'County',
                         'pcr_target_avg_conc': 'SC2_N_gc_g_dry_weight', 'hum_frac_mic_conc': 'PMMoV_gc_g_dry_weight',
                         'population_served': 'Population_Served'})

# Select only the columns of interest for both datasets
data_all_ww = data_all_ww[cols_CDPH]
data_all_ww_scan = data_all_ww_scan[cols_scan]

data_all_ww['Date'] = pd.to_datetime(data_all_ww['Date'])
data_all_ww_scan['Date'] = pd.to_datetime(data_all_ww_scan['Date'])



def get_data_SARS(save):
    """
        Get wastewater data related to SARS-CoV-2 genes across different plants in California.

        The function extracts data for specific genes related to COVID-19 from a wastewater dataset
        and aggregates it by plant and date.

        Args:
            data_all_ww (DataFrame): The wastewater dataset.

        Returns:
            DataFrame: Aggregated wastewater data for the specified genes.

    Note: In California, wastewater surveillance involves the detection of various genes, some targeting COVID-19 and
    its variants, while others target respiratory diseases such as Influenza and RSV. In this analysis, we focus
    on specific genes—N, n1, and n2. It's worth noting that 'N1' might sometimes appear as a mistake and is also
    found in the 'pcr_gene_target' column. The following genes are reported in the 'pcr_gene_target' column: ['n1', 'n2', 'N',
    'N1', 's', 'del143/145', 'lppa24s', 'InfA1', 'RSV-A and RSV-B combined', 'del69/70', 'G2R_G', 'InfB', 'G2R_WA', 'NoV GII', 'caur']
    In this analysis, we specifically focus on genes related to COVID-19. The 'target_gen' vector was manually created by identifying
    the genes used for COVID-19 detection.

    """
    # specify the group function to apply in each column in the data set
    agg_funcs = {'SC2_N_gc_g_dry_weight': 'mean', 'PMMoV_gc_g_dry_weight': 'mean',  'Population_Served': 'first',
                 'County': 'first', 'zipcode': 'first', 'pcr_gene_target': 'first'}

    #wwtp = data_all_ww.Plant.unique()
    data_ww_SARS = pd.DataFrame()

    plants_all = data_all_ww.Plant.unique()
    plants_all = plants_all[plants_all != 'Central Marin Sanitation Agency'] # This line is provisional. I'm cheking if I include the data from scan is better to take from cdph

    plant_n1 = ['LACSD_Jnt', 'LASAN_Hyp', 'SDPU_PtLom', 'Oxnard_WWTP']
    wwtp = [plant for plant in plants_all if plant not in plant_n1]


    for i in range (len(wwtp)):
        datap = data_all_ww[data_all_ww['Plant'] == wwtp[i]]
        if np.isin(['N'], datap.pcr_gene_target.unique()):
            datap = datap[datap['pcr_gene_target'] == 'N']
            datap['PMMoV_gc_g_dry_weight'] = datap['PMMoV_gc_g_dry_weight'].astype(float)
            datap['SC2_N_gc_g_dry_weight'] = datap['SC2_N_gc_g_dry_weight'].astype(float)
        elif np.isin(['n1', 'n2'], datap.pcr_gene_target.unique()).all():
            datap = datap[(datap['pcr_gene_target'] == 'n1') | (datap['pcr_gene_target'] == 'n2')]
            datap['PMMoV_gc_g_dry_weight'] = datap['PMMoV_gc_g_dry_weight'].astype(float)
            datap['SC2_N_gc_g_dry_weight'] = datap['SC2_N_gc_g_dry_weight'].astype(float)

        elif np.isin(['n1', 'N1'], datap.pcr_gene_target.unique()).all():
            datap= datap[(datap['pcr_gene_target'] == 'n1') | (datap['pcr_gene_target'] == 'N1')]
            datap['PMMoV_gc_g_dry_weight'] = datap['PMMoV_gc_g_dry_weight'].str.replace(',', '').astype(float)#.astype(float)
            datap['SC2_N_gc_g_dry_weight'] = datap['SC2_N_gc_g_dry_weight'].str.replace(',', '').astype(float)

        #  data_gn1N1= data_gn1N1[(data_gn1N1['pcr_gene_target'] == 'n1') | (data_gn1N1['pcr_gene_target'] == 'N1')]
        else:
            datap = datap[datap['pcr_gene_target'] == 'n1']
            datap['SC2_N_gc_g_dry_weight'] = datap['SC2_N_gc_g_dry_weight'].astype(float)
            datap['PMMoV_gc_g_dry_weight'] = datap['PMMoV_gc_g_dry_weight'].astype(float)
        datap = datap.groupby(['Plant', 'Date']).agg(agg_funcs).reset_index()
        data_ww_SARS = pd.concat([data_ww_SARS, datap])

    for i in range(len(plant_n1)):
        datap = data_all_ww[data_all_ww['Plant'] == plant_n1[i]]
        if plant_n1[i]== 'Oxnard_WWTP':
            datap = datap[(datap['pcr_gene_target'] == 'n1') | (datap['pcr_gene_target'] == 'N1')]
            #datap['SC2_N_gc_g_dry_weight'] = datap['SC2_N_gc_g_dry_weight'].astype(float)
            datap['SC2_N_gc_g_dry_weight'] = datap['SC2_N_gc_g_dry_weight'].str.replace(',', '').astype(float)
            datap['PMMoV_gc_g_dry_weight'] = datap['PMMoV_gc_g_dry_weight'].astype(float)
        else:
            datap = datap[datap['pcr_gene_target'] == 'n1']
            datap['SC2_N_gc_g_dry_weight'] = datap['SC2_N_gc_g_dry_weight'].astype(float)
            datap['PMMoV_gc_g_dry_weight'] = datap['PMMoV_gc_g_dry_weight'].astype(float)
            # data_ww_SARS.to_csv(save_path+"data_ww_plants_cdc.csv", index=False)
        datap = datap.groupby(['Plant', 'Date']).agg(agg_funcs).reset_index()
        data_ww_SARS = pd.concat([data_ww_SARS, datap])
    if save:
        data_ww_SARS.to_csv(save_path + "data_ww_CA_scan.csv", index=False)
    return data_ww_SARS



def joint_scan_cdph(save):
    """
    This function combines the datasets from the California Department of Public Health (CDPH)
    and scans for data pertaining to the cities of Merced, Modesto, and Davis. It's important
    to highlight that the CDPH dataset provides information up to December 31, 2022, for these
    specific locations, and it does not include the most recent updates beyond this date

    :param save:
    :return: dataframe
    """
    data_ww_SARS = get_data_SARS(save=False)

    cities_plants_scan = {'Merced': 'Merced Wastewater Treatment Plant', 'Modesto': 'Modesto Wastewater Primary Treatment Facility', 'Davis':'Davis_WWTP'}
    cities_scan = ['Merced', 'Modesto', 'Davis']
    #'Merced', Modesto, Davis  hay que pegarle los datos de scan
    agg_funcs = {'SC2_N_norm_PMMoV': 'mean', 'SC2_N_gc_g_dry_weight': 'mean', 'PMMoV_gc_g_dry_weight': 'mean',  'Population_Served': 'first', 'zipcode': 'first', 'Plant':'first'}

    for city in cities_scan:
        #city = cities_scan[i]
        wwtp = cities_plants_scan[city]

        data_scan_flt = data_all_ww_scan[data_all_ww_scan.City == city]
        if city=='Davis':
            data_scan_flt = data_all_ww_scan[data_all_ww_scan.Plant == 'City of Davis Wastewater Treatment Plant']

        data_scan_flt.loc[:, 'Plant'] = wwtp
        data_ww_SARS_flt = data_ww_SARS[data_ww_SARS['Plant'] == wwtp]
        data_scan_flt = data_scan_flt[data_scan_flt.Date > data_ww_SARS_flt.Date.iloc[-1]]
        data_scan_flt = data_scan_flt.groupby('Date').agg(agg_funcs).reset_index().reset_index()
        data_ww_scan_cdph = pd.concat([data_scan_flt, data_ww_SARS_flt]).sort_values('Date')

        data_ww_SARS= pd.concat([data_ww_SARS[data_ww_SARS['Plant'] != wwtp], data_ww_scan_cdph])

    data_scan_Marin = data_all_ww_scan[data_all_ww_scan.Plant == 'Central Marin Sanitation Agency']  # take form Marin of scan
    data_ww_SARS = pd.concat([data_ww_SARS, data_scan_Marin])
    if save:
        data_ww_SARS.to_csv(save_path + "data_ww_CA_cdph_scan.csv", index=False)
    return data_ww_SARS


def add_county_norm(data, cdph_scan):
    plants = data.Plant.unique()
    nomi = pgeocode.Nominatim('us')
    for plant in plants:
        zipcode = data[data.Plant == plant]['zipcode'].unique()[0]
        location = nomi.query_postal_code(str(int(zipcode)))
        if location['county_name'] == 'City and County of San Francisco':
            location['county_name'] = 'San Francisco'

        data.loc[data.Plant == plant, 'County'] = location['county_name']
        data.loc[data.Plant == plant, 'City'] = location['place_name']

    if cdph_scan:
        data.to_csv(save_path + "data_ww_CA_cdph_scan.csv", index=False)

    else:
        data.to_csv(save_path + "data_ww_CA_cdph.csv", index=False)

    return data


def add_county_name_scan(data):
    data = data[pd.notna(data['zipcode'])]
    plants = data.Plant.unique()
    nomi = pgeocode.Nominatim('us')
    for plant in plants:
        zipcode = data[data.Plant == plant]['zipcode'].unique()[0]
        location = nomi.query_postal_code(str(int(zipcode)))
        if location['county_name'] == 'City and County of San Francisco':
            location['county_name'] = 'San Francisco'
        data.loc[data.Plant == plant, 'County'] = location['county_name']
    return data



def Create_CA_ww_data_largest_pop(cdph_scan):
    if cdph_scan:
        data_cdph_scan = joint_scan_cdph(save=False)
        data_all_ww = add_county_norm(data=data_cdph_scan, cdph_scan=True)
        save_name = '_cdph_scan_'
    else:
        data_cdph = get_data_SARS(save=False)
        data_all_ww= add_county_norm(data=data_cdph, cdph_scan=False)
        save_name= '_cdph_'

    new_date = pd.to_datetime('2022-10-04')
    #data_all_ww['Date'] = pd.to_datetime(data_all_ww['Date'])

    data_all_ww_scan_c = add_county_name_scan(data=data_all_ww_scan)

    ### delete the largest plant for butte
    data_all_ww = data_all_ww[data_all_ww.Plant != 'Chico_WPCP']

    ######### Add san diego from scan
    data_all_ww = data_all_ww[data_all_ww.County!='San Diego']
    data_scan_SD = data_all_ww_scan_c[data_all_ww_scan_c.County == 'San Diego']  # take form Marin of scan
    data_all_ww = pd.concat([data_all_ww, data_scan_SD ])
    #######

    data_all_ww = data_all_ww[data_all_ww.Date >= new_date]
    counties_CA = data_all_ww.County.unique()
    # we selected the cities with the largest population served in each county
    plants = []
    for i, county in enumerate(counties_CA):
        data_county = data_all_ww[data_all_ww.County == county]
        if county=='Orange':
            pop_seved = data_county['Population_Served'][data_county['Population_Served'] != data_county['Population_Served'].max()].max()
        else:
            pop_seved = data_county["Population_Served"].max()
        plants.append(data_county.loc[data_county.Population_Served == pop_seved, "Plant"].iloc[0])

    data_ww = data_all_ww[data_all_ww.Plant.isin(plants)]
    data_ww.to_csv(save_path + "data_ww"+save_name+ "county.csv", index=False)

    return data_ww


data_ww_county = Create_CA_ww_data_largest_pop(cdph_scan=True)
counties_CA = data_ww_county.County.unique()

def data_county(save, donwload=donwload):
    if donwload:
        cases_all = pd.read_csv(
            "https://data.chhs.ca.gov/dataset/f333528b-4d38-4814-bebb-12db1f10f535/resource/046cdd2b-31e5-4d34-9ed3-b48cdbc4be7a/download/covid19cases_test.csv")
        cases_all.to_csv("data/" + 'covid19cases_test.csv')
    else:
        cases_all = pd.read_csv("data/" + 'covid19cases_test.csv')
    # Remove rows with nan dates
    cases_all.dropna(subset=["date"], inplace=True)
    cases_all = cases_all[cases_all['area'].isin(counties_CA)][['date', 'area', 'total_tests', 'positive_tests', 'population']]
    cases_all.date = pd.to_datetime(cases_all.date)


    cases_all['total_tests'] = np.where(cases_all['total_tests'] <= 0, np.nan, cases_all['total_tests'])
    cases_all['cases'] = np.where(cases_all['positive_tests'] < 0, 0, cases_all['positive_tests'])
    cases_all['cases'] = np.where(cases_all['total_tests'].isna(), np.nan, cases_all['cases'])

    cases_all['pos_rate'] = cases_all['cases'] / cases_all['total_tests']
    #cases_all['cases_pos_rate'] = cases_all['Cases'] / cases_all['total_tests']

    cases_all = cases_all.rename(columns={'area': 'County', 'date':'Date'})
    # cases_all = cases_all.drop(columns=['reported_tests', 'reported_cases'])

    if save:
        cases_all.to_csv("../output/" + "data_cases_county_CA.csv", index=False)
    return cases_all


data_cases_county = data_county(save=True, donwload=donwload)



def join_data_daily():
    Data_all = pd.DataFrame({})  # ,columns=var_names)

    for county in counties_CA:
        data_test_i = data_cases_county.loc[data_cases_county.County == county, ['Date', 'total_tests', 'cases', 'pos_rate']]
        #data_wasw_i = data_ww_county.loc[data_ww_county.County == county, ['Date', 'SC2_N_norm_PMMoV', "SC2_N_gc_g_dry_weight",'PMMoV_gc_g_dry_weight']]
        data_wasw_i = data_ww_county.loc[data_ww_county.County == county, ['Date', "SC2_N_gc_g_dry_weight", 'PMMoV_gc_g_dry_weight']]
        date_end = data_wasw_i["Date"].max()
        date_ini = data_wasw_i["Date"].min()
        if county=='Marin':
            data_ini= pd.to_datetime('2023-08-25')
        if county=='Madera':
            data_ini = pd.to_datetime('2023-05-26')

        data_test_i = data_test_i[(data_test_i["Date"] >= date_ini) & (data_test_i["Date"] <= date_end)]
        # data_ww = data_ww[(data_ww["date"] >= date_ini) & (data_ww["date"] <= date_end)]

        print(county)

        #data_wasw_i['N'] = data_wasw_i['SC2_N_gc_g_dry_weight']

        data_test_i.index = pd.to_datetime(data_test_i['Date'])
        data_wasw_i.index = pd.to_datetime(data_wasw_i['Date'])

        data_wasw_i['SC2_N_norm_PMMoV']= data_wasw_i["SC2_N_gc_g_dry_weight"]/data_wasw_i['PMMoV_gc_g_dry_weight']
        data_wasw_i['SC2_N_norm_PMMoV'] = data_wasw_i['SC2_N_norm_PMMoV'].replace(to_replace=0, value=data_wasw_i['SC2_N_norm_PMMoV'][
                                                                                            data_wasw_i[ 'SC2_N_norm_PMMoV'] != 0].min() / 2)  # Replace 0 by min
        test = data_test_i.groupby(pd.Grouper(freq="D")).mean()
        wwtp = data_wasw_i.groupby(pd.Grouper(freq="D")).mean()

        Data_full = pd.merge_ordered(test, wwtp, on=["Date"])
        Data_full["County"] = county

        Data_all = pd.concat([Data_all, Data_full], ignore_index=True)

    Data_all = Data_all[~Data_all.County.isin(del_counties)]
    Data_all.to_csv(save_path + "ww_cases_daily.csv", index=False)
    return Data_all


df = join_data_daily()



