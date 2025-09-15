Overview
Welcome to the Real Estate Demand Prediction Challenge! We are inviting the Kaggle community to participate in the first Kaggle competition that focuses on China's real estate market and help forecast monthly residential demand in China.

We look forward to seeing how you apply your data science expertise to help us shape the future of real estate!

Start

a month ago
Close
a month to go
Description
In China’s fast-evolving and highly dynamic housing market, accurately forecasting residential demand is vital for investment and development decisions. This competition challenges you to develop a machine learning model that predicts each sector's monthly sales for newly launched private residential projects, using historical transaction data, market conditions, and other relevant features.

Why Participate:
Real-World Impact: Your model will influence investment decisions and future development strategies, helping shape China's housing market.

Networking Opportunities: Engage with industry professionals and fellow Kaggle participants. Winners or excellent participants may be invited into the AI panel of a world renowned real estate group.

Prizes and Recognition: Win exciting prizes and showcase your skills to a global audience. This is your opportunity to tackle a real-world business problem and showcase your skills in data science and predictive modeling. Your insights will help shape the future of urban living in one of the world's most vibrant cities.

How to Get Started:
Review the provided data and competition details.

Build and refine your predictive model.

Submit your predictions.

We look forward to seeing how you apply your data science expertise to help us shape the future of real estate!

Evaluation
Submissions will be evaluated using a custom two-stage metric based on the absolute percentage error between predicted and actual values. The custom score is computed from a scaled MAPE with two stages.

First stage:

If over 30% of the submitted samples have absolute percentage errors exceeding 100%, a score of 0 is immediately given.

score_0.png

Second stage:

Otherwise, an MAPE is calculated with the samples that have absolute percentage errors less than or equal to 1. Then the MAPE is divided by the fraction of absolute percentage errors less than or equal to 1. Finally, the score is 1 minus the scaled MAPE.

score_1.png

Submission File
The submission file must contain exactly these two columns: id and new_house_transaction_amount. Each id is formatted as a month in the form of %Y %b and a sector id in the form of sector n joined by an underscore. For each id in the test set, you must predict the total new house transaction amount in ten thousand Chinese Yuan for the specific month and sector. The total new house transaction amount is equal to total new house transaction area times transaction price per area. The file should contain a header and have the following format:

id,new_house_transaction_amount
2024 Aug_sector 1,49921.00868
2024 Aug_sector 2,92356.22985
2024 Aug_sector 3,480269.7819
etc.
Submission Requirements
Preserve row order: Maintain the exact same row order as in test.csv
Exact Columns: The submission file should contain the exact columns as in test.csv
Prizes
Top three participants/teams will win prizes with potential bonus.

Total Prizes Available: $10,000 USD

First Prize(1 winner) - $2,500 USD or $5,000 USD with bonus
Second Prize(1 winner) - $1,500 USD or $3,000 USD with bonus
Third Prize(1 winner) - $1,000 USD or $2,000 USD with bonus
Bonus Performance Threshold
Winners who achieve final score ≥ 0.75 receive a bonus by doubling their prize amount.

Winner Selection
Additional Verification: Private leaderboard metrics are not final. Organizers will request code from top-10 participants for final evaluation. Winning models may be incorporated into a real estate company's proprietary forecasting system for internal use.
New Data Testing: Organizers reserve the right to test candidate solutions on unpublished 2025 data.

Dataset Description
Participants are expected to use data in the train folder which contains all files for train set to predict monthly price of new house transactions for each sector.

Predict Target
The target is in train/new_house_transactions.csv file under column amount_new_house_transaction. Please note that not all sectors and months are included in the file. For example, there is no data in the file for 2019 Jan_sector 3, which means participants are expected to predict a value of 0.

month,sector,num_new_house_transactions,area_new_house_transactions,amount_new_house_transaction,...
2019 Jan_sector 1,52,4906,28184,...
2019 Jan_sector 2,145,15933,17747,...
2019 Jan_sector 4,6,725,28004,...
2019 Jan_sector 5,2,212,37432,...
2019 Jan_sector 6,5,773,15992,...
Avoid Data Leakage
When predicting for a certain month, participants are not supposed to use any data corresponding to a future time period relative to said month.

Files
train/*.csv - the training set
test.csv - the test set
sample_submission.csv - a sample submission file in the correct format
Columns
Unless specified otherwise, all appearances of yuan in this section stand for the Chinese yuan, the official currency of China.

train/pre_owned_house_transactions.csv

month: The month of the transaction.
sector: The specific geographic sector where the transaction occurred.
area_pre_owned_house_transactions: The total area of pre-owned house transactions in square meters.
amount_pre_owned_house_transactions: The total monetary value of pre-owned house transactions in 10,000 yuan.
num_pre_owned_house_transactions: The total number of pre-owned house transactions.
price_pre_owned_house_transactions: The average price per square meter of pre-owned house transactions in yuan.
train/pre_owned_house_transactions_nearby_sectors.csv

month: The month of the transaction.
sector: The specific geographic sector of interest.
area_pre_owned_house_transactions_nearby_sectors: The total area of pre-owned house transactions in nearby sectors in square meters.
amount_pre_owned_house_transactions_nearby_sectors: The total monetary value of pre-owned house transactions in nearby sectors in 10,000 yuan.
num_pre_owned_house_transactions_nearby_sectors: The total number of pre-owned house transactions in nearby sectors.
price_pre_owned_house_transactions_nearby_sectors: The average price per square meter of pre-owned house transactions in nearby sectors in yuan.
train/land_transactions.csv

month: The month of the transaction.
sector: The specific geographic sector where the land transaction occurred.
num_land_transactions: The total number of land transactions.
construction_area: The total area of land designated for construction in square meters.
planned_building_area: The total planned building area on the transacted land in square meters.
transaction_amount: The total monetary value of land transactions in 10,000 yuan.
train/land_transactions_nearby_sectors.csv

month: The month of the transaction.
sector: The specific geographic sector of interest.
num_land_transactions_nearby_sectors: The total number of land transactions in nearby sectors.
construction_area_nearby_sectors: The total area of land designated for construction in nearby sectors in square meters.
planned_building_area_nearby_sectors: The total planned building area on transacted land in nearby sectors in square meters.
transaction_amount_nearby_sectors: The total monetary value of land transactions in nearby sectors in 10,000 yuan.
train/new_house_transactions.csv

month: The month of the transaction.
sector: The specific geographic sector where the new house transaction occurred.
num_new_house_transactions: The total number of new house transactions.
area_new_house_transactions: The total area of new house transactions in square meters.
price_new_house_transactions: The average price per square meter of new house transactions in yuan.
amount_new_house_transactions: The total monetary value of new house transactions in 10,000 yuan.
area_per_unit_new_house_transactions: The average area per new house transaction unit in square meters per unit.
total_price_per_unit_new_house_transactions: The average total price per new house transaction unit in 10,000 yuan per unit.
num_new_house_available_for_sale: The total number of new houses available for sale.
area_new_house_available_for_sale: The total area of new houses available for sale in square meters.
period_new_house_sell_through: The estimated time in months to sell all available new houses.
train/new_house_transactions_nearby_sectors.csv

month: The month of the transaction.
sector: The specific geographic sector of interest.
num_new_house_transactions_nearby_sectors: The total number of new house transactions in nearby sectors.
area_new_house_transactions_nearby_sectors: The total area of new house transactions in nearby sectors in square meters.
price_new_house_transactions_nearby_sectors: The average price per square meter of new house transactions in nearby sectors in yuan.
amount_new_house_transactions_nearby_sectors: The total monetary value of new house transactions in nearby sectors in 10,000 yuan.
area_per_unit_new_house_transactions_nearby_sectors: The average area per new house transaction unit in nearby sectors in square meters per unit.
total_price_per_unit_new_house_transactions_nearby_sectors: The average total price per new house transaction unit in nearby sectors in 10,000 yuan per unit.
num_new_house_available_for_sale_nearby_sectors: The total number of new houses available for sale in nearby sectors.
area_new_house_available_for_sale_nearby_sectors: The total area of new houses available for sale in nearby sectors in square meters.
period_new_house_sell_through_nearby_sectors: The estimated time in months to sell all available new houses in nearby sectors.
train/sector_POI.csv

sector: The specific geographic sector.
sector_coverage: The geographical extent or area covered by the sector.
population_scale: The general size of the population within the sector.
residential_area: The presence or extent of residential zones within the sector.
office_building: The presence or extent of office buildings within the sector.
commercial_area: The presence or extent of commercial zones within the sector.
resident_population: The number of people residing in the sector.
office_population: The number of people working in offices within the sector.
number_of_shops: The total count of shops in the sector.
catering: The number or density of catering establishments.
retail: The number or density of retail establishments.
hotel: The number or density of hotel establishments.
transportation_station: The number or density of transportation stations.
education: The number or density of educational facilities.
leisure_and_entertainment: The number or density of leisure and entertainment venues.
bus_station_cnt: The count of bus stations.
subway_station_cnt: The count of subway stations.
rentable_shops: The number of shops available for rent.
surrounding_housing_average_price: The average price of housing in the surrounding area.
surrounding_shop_average_rent: The average rent of shops in the surrounding area.
leisure_entertainment_entertainment_venue_game_arcade: The number or density of game arcades.
leisure_entertainment_entertainment_venue_party_house: The number or density of party houses.
leisure_entertainment_cultural_venue_cultural_palace: The number or density of cultural palaces.
office_building_industrial_building_industrial_building: The number or density of industrial buildings used as office spaces.
education_training_school_education_middle_school: The number or density of middle schools.
education_training_school_education_primary_school: The number or density of primary schools.
education_training_school_education_kindergarten: The number or density of kindergartens.
education_training_school_education_research_institution: The number or density of research institutions.
medical_health: The number or density of general medical and health facilities.
medical_health_specialty_hospital: The number or density of specialty hospitals.
medical_health_tcm_hospital: The number or density of Traditional Chinese Medicine (TCM) hospitals.
medical_health_physical_examination_institution: The number or density of physical examination institutions.
medical_health_veterinary_station: The number or density of veterinary stations.
medical_health_pharmaceutical_healthcare: The number or density of pharmaceutical healthcare providers.
medical_health_rehabilitation_institution: The number or density of rehabilitation institutions.
medical_health_first_aid_center: The number or density of first aid centers.
medical_health_blood_donation_station: The number or density of blood donation stations.
medical_health_disease_prevention_institution: The number or density of disease prevention institutions.
medical_health_general_hospital: The number or density of general hospitals.
medical_health_clinic: The number or density of clinics.
transportation_facilities_service_bus_station: The presence or density of bus stations.
transportation_facilities_service_subway_station: The presence or density of subway stations.
transportation_facilities_service_airport_related: The presence or density of airport-related facilities.
transportation_facilities_service_port_terminal: The presence or density of port or terminal facilities.
transportation_facilities_service_train_station: The presence or density of train stations.
transportation_facilities_service_light_rail_station: The presence or density of light rail stations.
transportation_facilities_service_long_distance_bus_station: The presence or density of long-distance bus stations.
number_of_leisure_and_entertainment_stores: The count of leisure and entertainment stores.
number_of_other_stores: The count of miscellaneous other stores.
number_of_other_anchor_stores: The count of other major or anchor stores.
number_of_home_appliance_stores: The count of home appliance stores.
number_of_skincare_cosmetics_stores: The count of skincare and cosmetics stores.
number_of_fashion_stores: The count of fashion stores.
number_of_service_stores: The count of service-oriented stores.
number_of_jewelry_stores: The count of jewelry stores.
number_of_lifestyle_leisure_stores: The count of lifestyle and leisure stores.
number_of_supermarket_convenience_stores: The count of supermarkets and convenience stores.
number_of_catering_food_stores: The count of catering and food stores.
number_of_residential_commercial: The count of commercial establishments within residential areas.
number_of_office_building_commercial: The count of commercial establishments within office buildings.
number_of_commercial_buildings: The count of dedicated commercial buildings.
number_of_hypermarkets: The count of hypermarkets.
number_of_department_stores: The count of department stores.
number_of_shopping_centers: The count of shopping centers.
number_of_hotel_commercial: The count of commercial establishments within hotels.
number_of_third_tier_shopping_malls_in_business_district: The count of third-tier shopping malls within the business district.
number_of_second_tier_shopping_malls_in_business_district: The count of second-tier shopping malls within the business district.
number_of_city_winner_malls: The count of high-performing "city winner" malls.
number_of_shopping_malls_with_street_facing_shops: The count of shopping malls featuring street-facing shops.
number_of_unranked_malls: The count of shopping malls without a specific ranking.
number_of_community_malls: The count of community-focused malls.
number_of_community_winner_malls: The count of high-performing "community winner" malls.
number_of_key_focus_malls: The count of shopping malls identified for key focus.
population_scale_dense: The density of the population scale within the sector.
residential_area_dense: The density of residential areas within the sector.
office_building_dense: The density of office buildings within the sector.
commercial_area_dense: The density of commercial areas within the sector.
resident_population_dense: The density of the resident population within the sector.
office_population_dense: The density of the office population within the sector.
number_of_shops_dense: The density of shops within the sector.
catering_dense: The density of catering establishments within the sector.
retail_dense: The density of retail establishments within the sector.
hotel_dense: The density of hotel establishments within the sector.
transportation_station_dense: The density of transportation stations within the sector.
education_dense: The density of educational facilities within the sector.
leisure_and_entertainment_dense: The density of leisure and entertainment venues within the sector.
bus_station_cnt_dense: The density of bus stations.
subway_station_cnt_dense: The density of subway stations.
rentable_shops_dense: The density of rentable shops.
leisure_entertainment_stores_dense: The density of leisure and entertainment stores.
other_stores_dense: The density of miscellaneous other stores.
other_anchor_stores_dense: The density of other major or anchor stores.
home_appliance_stores_dense: The density of home appliance stores.
skincare_cosmetics_stores_dense: The density of skincare and cosmetics stores.
fashion_stores_dense: The density of fashion stores.
service_stores_dense: The density of service-oriented stores.
jewelry_stores_dense: The density of jewelry stores.
lifestyle_leisure_stores_dense: The density of lifestyle and leisure stores.
supermarket_convenience_stores_dense: The density of supermarkets and convenience stores.
catering_food_stores_dense: The density of catering and food stores.
residential_commercial_dense: The density of commercial establishments within residential areas.
office_building_commercial_dense: The density of commercial establishments within office buildings.
commercial_buildings_dense: The density of dedicated commercial buildings.
hypermarkets_dense: The density of hypermarkets.
department_stores_dense: The density of department stores.
shopping_centers_dense: The density of shopping centers.
hotel_commercial_dense: The density of commercial establishments within hotels.
third_tier_shopping_malls_in_business_district_dense: The density of third-tier shopping malls within the business district.
second_tier_shopping_malls_in_business_district_dense: The density of second-tier shopping malls within the business district.
city_winner_malls_dense: The density of high-performing "city winner" malls.
shopping_malls_with_street_facing_shops_dense: The density of shopping malls featuring street-facing shops.
unranked_malls_dense: The density of shopping malls without a specific ranking.
community_malls_dense: The density of community-focused malls.
community_winner_malls_dense: The density of high-performing "community winner" malls.
key_focus_malls_dense: The density of shopping malls identified for key focus.
transportation_facilities_service_bus_station_dense: The density of bus stations.
transportation_facilities_service_subway_station_dense: The density of subway stations.
transportation_facilities_service_airport_related_dense: The density of airport-related facilities.
transportation_facilities_service_port_terminal_dense: The density of port or terminal facilities.
transportation_facilities_service_train_station_dense: The density of train stations.
transportation_facilities_service_light_rail_station_dense: The density of light rail stations.
transportation_facilities_service_long_distance_bus_station_dense: The density of long-distance bus stations.
leisure_entertainment_entertainment_venue_game_arcade_dense: The density of game arcades.
leisure_entertainment_entertainment_venue_party_house_dense: The density of party houses.
leisure_entertainment_cultural_venue_cultural_palace_dense: The density of cultural palaces.
office_building_industrial_building_industrial_building_dense: The density of industrial buildings used as office spaces.
medical_health_dense: The density of general medical and health facilities.
medical_health_specialty_hospital_dense: The density of specialty hospitals.
medical_health_tcm_hospital_dense: The density of Traditional Chinese Medicine (TCM) hospitals.
medical_health_physical_examination_institution_dense: The density of physical examination institutions.
medical_health_veterinary_station_dense: The density of veterinary stations.
medical_health_pharmaceutical_healthcare_dense: The density of pharmaceutical healthcare providers.
medical_health_rehabilitation_institution_dense: The density of rehabilitation institutions.
medical_health_first_aid_center_dense: The density of first aid centers.
medical_health_blood_donation_station_dense: The density of blood donation stations.
medical_health_disease_prevention_institution_dense: The density of disease prevention institutions.
medical_health_general_hospital_dense: The density of general hospitals.
medical_health_clinic_dense: The density of clinics.
education_training_school_education_middle_school_dense: The density of middle schools.
education_training_school_education_primary_school_dense: The density of primary schools.
education_training_school_education_kindergarten_dense: The density of kindergartens.
education_training_school_education_research_institution_dense: The density of research institutions.
train/city_search_index.csv

month: The month the search data was recorded.
keyword: The specific search term.
source: The origin or platform of the search data.
search_volume: The total number of searches for the keyword.
train/city_indexes.csv

city_indicator_data_year: The year to which the city indicator data pertains.
year_end_registered_population_10k: The registered population at year-end in tens of thousands.
total_households_10k: The total number of households in tens of thousands.
year_end_resident_population_10k: The permanent resident population at year-end in tens of thousands.
year_end_total_employed_population_10k: The total employed population at year-end in tens of thousands.
year_end_urban_non_private_employees_10k: The number of urban non-private unit employees at year-end in tens of thousands.
private_individual_and_other_employees_10k: The number of private, individual, and other employees in tens of thousands.
private_individual_ratio: The proportion of private and individual employees.
national_year_end_total_population_10k: The national total population at year-end in tens of thousands.
resident_registered_ratio: The ratio of permanent residents to registered population.
under_18_10k: The population under 18 years old in tens of thousands.
18_60_years_10k: The population aged 18 to 60 years old in tens of thousands.
over_60_years_10k: The population over 60 years old in tens of thousands.
total: The total population count.
under_18_percent: The percentage of the population under 18 years old.
18_60_years_percent: The percentage of the population aged 18 to 60 years old.
over_60_years_percent: The percentage of the population over 60 years old.
gdp_100m: The Gross Domestic Product (GDP) in 100 million yuan.
primary_industry_100m: The output value of the primary industry in 100 million yuan.
secondary_industry_100m: The output value of the secondary industry in 100 million yuan.
tertiary_industry_100m: The output value of the tertiary industry in 100 million yuan.
gdp_per_capita_yuan: The GDP per capita in yuan.
national_gdp_100m: The national GDP in 100 million yuan.
national_economic_primacy: An indicator of the city's economic dominance compared to the nation.
national_population_share: The city's share of the national population.
gdp_population_ratio: The ratio of the city's GDP primacy to its national population share.
secondary_industry_development_gdp_share: The share of the secondary industry in the GDP, indicating its development.
tertiary_industry_development_gdp_share: The share of the tertiary industry in the GDP, indicating its development.
employed_population: The total number of employed individuals.
primary_industry_percent: The percentage of the employed population in the primary industry.
secondary_industry_percent: The percentage of the employed population in the secondary industry.
tertiary_industry_percent: The percentage of the employed population in the tertiary industry.
white_collar_service_vs_blue_collar_manufacturing_ratio: The ratio of white-collar (service industry) to blue-collar (manufacturing industry) population.
general_public_budget_revenue_100m: The general public budget revenue in 100 million yuan.
personal_income_tax_100m: The personal income tax collected in 100 million yuan.
per_capita_personal_income_tax_yuan: The per capita personal income tax in yuan.
general_public_budget_expenditure_100m: The general public budget expenditure in 100 million yuan.
total_retail_sales_of_consumer_goods_100m: The total retail sales of consumer goods in 100 million yuan.
retail_sales_growth_rate: The growth rate of retail sales.
urban_consumer_price_index_previous_year_100: The urban consumer price index, with the previous year set as 100.
annual_average_wage_urban_non_private_employees_yuan: The annual average wage of urban non-private unit employees in yuan.
annual_average_wage_urban_non_private_on_duty_employees_yuan: The annual average wage of urban non-private on-duty employees in yuan.
per_capita_disposable_income_absolute_yuan: The absolute value of per capita disposable income in yuan.
per_capita_disposable_income_index_previous_year_100: The per capita disposable income index, with the previous year set as 100.
engel_coefficient: The Engel coefficient, indicating the proportion of income spent on food.
per_capita_housing_area_sqm: The per capita housing area in square meters.
number_of_universities: The total count of universities and colleges.
university_students_10k: The number of university students in tens of thousands.
number_of_middle_schools: The total count of middle schools.
middle_school_students_10k: The number of middle school students in tens of thousands.
number_of_primary_schools: The total count of primary schools.
primary_school_students_10k: The number of primary school students in tens of thousands.
number_of_kindergartens: The total count of kindergartens.
kindergarten_students_10k: The number of kindergarten students in tens of thousands.
hospitals_health_centers: The total count of hospitals and health centers.
hospital_beds_10k: The number of hospital beds in tens of thousands.
health_technical_personnel_10k: The number of health technical personnel in tens of thousands.
doctors_10k: The number of doctors in tens of thousands.
road_length_km: The total length of roads in kilometers.
road_area_10k_sqm: The total area of roads in 10,000 square meters.
per_capita_urban_road_area_sqm: The per capita urban road area in square meters.
number_of_operating_bus_lines: The total count of operating bus lines.
operating_bus_line_length_km: The total length of operating bus lines in kilometers.
internet_broadband_access_subscribers_10k: The number of internet broadband access subscribers in tens of thousands.
internet_broadband_access_ratio: The ratio of internet broadband access.
number_of_industrial_enterprises_above_designated_size: The total count of industrial enterprises above a designated size.
total_current_assets_10k: The total current assets in 10,000 yuan.
total_fixed_assets_10k: The total fixed assets in 10,000 yuan.
main_business_taxes_and_surcharges_10k: The main business taxes and surcharges in 10,000 yuan.
total_fixed_asset_investment_10k: The total fixed asset investment in 10,000 yuan.
real_estate_development_investment_completed_10k: The completed real estate development investment in 10,000 yuan.
residential_development_investment_completed_10k: The completed residential development investment in 10,000 yuan.
science_expenditure_10k: The expenditure on science in 10,000 yuan.
education_expenditure_10k: The expenditure on education in 10,000 yuan.

EXAMPLE SOLUTIONS:


import pandas as pd

# Read 2 submission files
sub1 = pd.read_csv("/kaggle/input/enhancing-with-weight-decay-from-geometric-mean/submission.csv")
sub2 = pd.read_csv("/kaggle/input/simple-seasonality-bump/submission.csv")

# Check if ids match
assert all(sub1["id"] == sub2["id"]), "IDs do not match between the two files!"

# Ensemble (you can adjust the weights if needed)
sub_ens = sub1.copy()
sub_ens["new_house_transaction_amount"] = (
    0.35 * sub1["new_house_transaction_amount"] +
    0.55 * sub2["new_house_transaction_amount"]
)

# Save to a new file
sub_ens.to_csv("submission.csv", index=False)

print("✅ Ensemble submission.csv file created successfully")

Real Estate Demand Prediction: Explained baseline
This notebook shows a baseline model for the competition and its cross-validation.

Reference:

Kaggle competition
def custom_score(y_true, y_pred, eps=1e-12):
    """Scoring function of the competition as defined on the competition overview page.
    
    Parameters:
    -----------
    y_true : array-like
    y_pred : array-like
    eps : float, optional (exact value doesn't matter)

    Return value:
    -------------
    dict with keys 'score', 'good_rate' and 'str'
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        raise ValueError('empty array')

    if (y_true < 0).any():
        raise ValueError('negative y_true')

    if (~ np.isfinite(y_pred)).any():
        raise ValueError('infinite y_pred')

    ape = np.abs((y_true - y_pred) / np.maximum(y_true, eps))

    good_mask = ape <= 1.0
    good_rate = good_mask.mean()
    if good_rate < 0.7:
        return {'score': 0, 'good_rate': good_rate, 'str': f"{Fore.RED}score={0:.3f} {good_rate=:.3f}{Style.RESET_ALL}"}

    good_ape = ape[good_mask]
    mape = np.mean(good_ape)

    scaled_mape = mape / good_rate
    score = 1 - scaled_mape
    # score = max(0.0, score)
    return {'score': score, 'good_rate': good_rate, 'str': f"{score=:.3f} {good_rate=:.3f}"}
# We read all the data although this baseline notebook ignores most of it
# We convert the string-encoded months to integer values (time is 0..66 for train and 67..78 for test)

ci = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_indexes.csv') # one row per year
csi = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/city_search_index.csv') # several rows per training month
sp = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/sector_POI.csv') # at most one row per sector

train_lt = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions.csv')
train_ltns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/land_transactions_nearby_sectors.csv')
train_pht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions.csv')
train_phtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/pre_owned_house_transactions_nearby_sectors.csv')
train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
train_nhtns = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions_nearby_sectors.csv')
test = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')

month_codes = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}

test_id = test.id.str.split('_', expand=True)
test['month'] = test_id[0]
test['sector'] = test_id[1]
del test_id

for df in [train_lt, train_ltns, train_pht, train_phtns, train_nht, train_nhtns, csi, sp, test]:
    if df is not csi:
        df['sector_id'] = df.sector.str.slice(7, None).astype(int)
        # print(df.sector_id.min(), df.sector_id.max(), len(np.unique(df.sector_id)), len(df))
    if df is not sp:
        df['year'] = df.month.str.slice(0, 4).astype(int)
        df['month'] = df.month.str.slice(5, None).map(month_codes)
        df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1 # min=0, max=66
        print(df['time'].min(), df['time'].max())
0 66
0 66
0 66
0 66
0 66
0 66
0 66
67 78
Initial observations
This is a time series competition: The training dataset comprises Jan 2019 through Jul 2024 (67 months). The test period is Aug 2024 through Jul 2025 (12 months).

The prediction target is amount_new_house_transactions in train/new_house_transactions.csv.

The data are organized into 96 sectors. The sectors have widely varying sizes and different missing value patterns. Sector 95 has no new house transations during the training period.

The test dataframe is the cartesian product of the 96 sectors and the 12 test months. This means that we must predict amount_new_house_transactions for all 96 sectors and all 12 test months. In other words: We have to extrapolate 96 time series for 12 time steps each.

We now extract the target variable amount_new_house_transactions into a 2d array. Every column of the array is a time series, and the competition tasks basically consists of extrapolating to the next 12 rows of this array:

amount_new_house_transactions = train_nht.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack()
# Missing values must be filled with zero:
amount_new_house_transactions = amount_new_house_transactions.fillna(0)
# We add sector 95, which has no transactions during the training period:
amount_new_house_transactions[95] = 0
amount_new_house_transactions = amount_new_house_transactions[np.arange(1, 97)]
amount_new_house_transactions.astype(int)
sector_id	1	2	3	4	5	6	7	8	9	10	...	87	88	89	90	91	92	93	94	95	96
time																					
0	13827	28277	0	1424	792	607	39326	10454	4170	11043	...	0	37312	9676	1795	14989	26427	16539	70238	0	0
1	8802	12868	0	1522	409	603	13707	3015	1318	9916	...	0	14923	5709	0	4185	15702	12333	43823	0	0
2	23283	18694	0	1779	833	1024	16279	3602	30631	10852	...	263	57859	25247	0	9404	28988	26310	116638	0	809
3	26626	15460	0	663	0	3471	17472	6238	27396	3387	...	0	17610	14016	0	6764	37055	34804	146907	0	535
4	8649	20565	0	1387	0	4863	12227	3597	10354	3140	...	0	18462	4510	2279	7866	37666	52138	108483	0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
62	4605	7552	15716	71227	1847	12306	7058	7474	44942	44429	...	0	4187	3559	9023	29480	38262	13316	21061	0	0
63	3622	3155	3248	64991	2712	19563	5948	6278	10381	52813	...	0	3111	2304	9620	11058	28710	13961	18301	0	0
64	4229	3833	1013	71032	500	17662	5503	2653	6260	54442	...	0	3594	0	7302	19585	28258	15912	18053	0	0
65	19445	16514	8205	98927	7173	19212	7392	2668	8505	77559	...	0	4703	2759	4538	54214	30980	15043	16515	0	0
66	9295	2197	9428	83402	1575	12505	10583	4549	12885	42217	...	0	3774	0	7631	32450	30804	22335	13389	0	561
67 rows × 96 columns

A diagram of the total amounts (summed over all 96 sectors) shows that the time series has no obvious trend, but distinctive peaks every December:


Baseline prediction
Our baseline model works as follows:

If in the last six months of the training data for a sector there is any month with a zero amount, we predict zero. Otherwise there would be a high risk that y_true is zero, y_pred is nonzero, and the contribution to the MAPE would be huge.
If the last six months all had nonzero amounts, we predict the gemoetric mean of the last six training months for the whole test period.
For the moment, we ignore the yearly seasonality.
... and we ignore all other features of the dataset.
To cross-validate the model, we use a TimeSeriesSplit:

t1 = 6 # months for geometric mean
t2 = 6 # months which must be nonzero
cv = TimeSeriesSplit(n_splits=4, test_size=12)
true, oof = [], []
for fold, (idx_tr, idx_va) in enumerate(cv.split(amount_new_house_transactions)):
    print(f"# Fold {fold}: train on months {idx_tr.min()}..{idx_tr.max()}, validate on months {idx_va.min()}..{idx_va.max()}")
    a_tr = amount_new_house_transactions.iloc[idx_tr]
    a_va = amount_new_house_transactions.iloc[idx_va]

    a_pred = pd.DataFrame(
        {time: np.exp(np.log(a_tr.tail(t1)).mean(axis=0)) for time in idx_va}
    ).T
    a_pred.loc[:, a_tr.tail(t2).min(axis=0) == 0] = 0
    a_pred.index.rename('time', inplace=True)
    # display(a_pred.astype(int))
    print(f"# Fold {fold}: {custom_score(a_va, a_pred)['str']}\n")
    true.append(a_va)
    oof.append(a_pred)

print(f"# Overall {custom_score(pd.concat(true), pd.concat(oof))['str']} {t1=} {t2=}\n")
# Fold 0: train on months 0..18, validate on months 19..30
# Fold 0: score=0.391 good_rate=0.941

# Fold 1: train on months 0..30, validate on months 31..42
# Fold 1: score=0.440 good_rate=0.759

# Fold 2: train on months 0..42, validate on months 43..54
# Fold 2: score=0.479 good_rate=0.840

# Fold 3: train on months 0..54, validate on months 55..66
# Fold 3: score=0.511 good_rate=0.803

# Overall score=0.447 good_rate=0.836 t1=6 t2=6
# Fold 0: train on months 0..18, validate on months 19..30
# Fold 0: score=0.391 good_rate=0.941

# Fold 1: train on months 0..30, validate on months 31..42
# Fold 1: score=0.440 good_rate=0.759

# Fold 2: train on months 0..42, validate on months 43..54
# Fold 2: score=0.479 good_rate=0.840

# Fold 3: train on months 0..54, validate on months 55..66
# Fold 3: score=0.511 good_rate=0.803

# Overall score=0.447 good_rate=0.836 t1=6 t2=6

Submission
a_tr = amount_new_house_transactions
a_pred = pd.DataFrame(
    {time: a_tr.tail(t1).mean(axis=0) for time in np.arange(67, 79)}
).T
a_pred.loc[:, a_tr.tail(t2).min(axis=0) == 0] = 0
a_pred.index.rename('time', inplace=True)
display(a_pred.astype(int))
sector_id	1	2	3	4	5	6	7	8	9	10	...	87	88	89	90	91	92	93	94	95	96
time																					
67	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
68	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
69	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
70	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
71	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
72	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
73	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
74	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
75	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
76	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
77	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
78	7912	6199	7561	69254	2399	16158	6529	4202	16586	47686	...	0	3284	0	7177	26101	28374	15521	16183	0	0
12 rows × 96 columns

test['new_house_transaction_amount'] = a_pred.T.unstack().values

test[['id', 'new_house_transaction_amount']].to_csv('submission.csv', index=False)
!head submission.csv
id,new_house_transaction_amount
2024 Aug_sector 1,7912.346666666665
2024 Aug_sector 2,6199.68
2024 Aug_sector 3,7561.94
2024 Aug_sector 4,69254.1
2024 Aug_sector 5,2399.6983333333333
2024 Aug_sector 6,16158.851666666667
2024 Aug_sector 7,6529.978333333333
2024 Aug_sector 8,4202.766666666667
2024 Aug_sector 9,16586.418333333335
Conclusion
The predictive model of this notebook will have little business value for the competition host:
Predicting that future transaction amounts will correspond to the average of the previous six months is naive. You need neither data science nor a computer to do that.
Predicting zero if there were any months with zero transactions in the past doesn't make much sense for the business. This part of the model is an artefact of the chosen metric (MAPE). Wikipedia says: MAPE should be used with extreme caution in forecasting, because small actuals (target labels) can lead to highly inflated MAPE scores.

The training dataset is small. There are too few samples for the many features in ci and sp. Stick to simple models or you'll overfit!

TOP 5 SCORES

1
LIH.YUN
0.61076
63
2d
2
Recep Barkın Topcu
0.60750
82
6d
3
Kirill Tsukanov
0.60717
80
10d
4
mirko ferretti
0.60366
101
3d
5
Toan Nguyen Mau
0.59984
37
11d
