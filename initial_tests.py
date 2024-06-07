from river import linear_model
from river.metrics import MSE
from river.evaluate import progressive_val_score
from river import stream
from river import forest

model =   linear_model.LogisticRegression()#insert Classifier/algorithm here

# stream = FileStream("../power_plant/MPHA-docs-data.csv")


# TmStamp,RecNum,batt_volt,mean_wind_speed,mean_wind_direction,std_wind_dir,Max_Gust_Min,Max_Gust_Hr,Rain_mm,Barametric_Avg,Air_Temp_Avg,RH_Avg,SlrW_Avg,SlrMJ_Tot,in_bytes_str,Dew_Point_Avg
params = {
    'converters': {'rating': float},
    'parse_dates': {'year': '%Y'}
}

def parse_float(x):
    try:
        return float(x)
    except ValueError:
        return None


# https://riverml.xyz/latest/api/stream/iter-csv/
dataset = stream.iter_csv('../power_plant/test_data.csv', 
                        target="SlrW_Avg",
                        converters={
                            'Rain_mm':int,
                            'batt_volt': parse_float,
                            'mean_wind_speed': parse_float,
                            'mean_wind_direction': parse_float,
                            'std_wind_dir': parse_float,
                            'Max_Gust_Min': parse_float,
                            'Max_Gust_Hr': parse_float,
                            'Barametric_Avg': parse_float,
                            'Air_Temp_Avg': parse_float,
                            'RH_Avg': parse_float,
                            'in_bytes_str': parse_float,
                            'Dew_Point_Avg': parse_float,
                            'SlrW_Avg': parse_float,
                        },
                        drop_nones= True,
                        drop=["TmStamp","RecNum","SlrMJ_Tot"] )

# for x, y in dataset:
#     print(x, y)

print("Loaded data")

model = forest.ARFRegressor()

score = progressive_val_score(dataset=dataset,
                        model=model,
                        metric=MSE(),
                        show_memory=True,
                        show_time=True,
                        print_every=50000)  #gives us the current Metric every X datapoint

print(score)



