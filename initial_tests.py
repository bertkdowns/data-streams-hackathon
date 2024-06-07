from river import linear_model
from river.metrics import Accuracy
from river.evaluate import progressive_val_score
from river import stream

model =   linear_model.LogisticRegression()#insert Classifier/algorithm here

# stream = FileStream("../power_plant/MPHA-docs-data.csv")


# TmStamp,RecNum,batt_volt,mean_wind_speed,mean_wind_direction,std_wind_dir,Max_Gust_Min,Max_Gust_Hr,Rain_mm,Barametric_Avg,Air_Temp_Avg,RH_Avg,SlrW_Avg,SlrMJ_Tot,in_bytes_str,Dew_Point_Avg
params = {
    'converters': {'rating': float},
    'parse_dates': {'year': '%Y'}
}

# https://riverml.xyz/latest/api/stream/iter-csv/
dataset = stream.iter_csv('../power_plant/MHPA-docs-data.csv', 
                        target="SlrW_Avg",
                        converters={
                            'Rain_mm':int
                        },
                        drop_nones= True,
                        drop=["TmStamp","RecNum","SlrMJ_Tot"] )

for x, y in dataset:
    print(x, y)