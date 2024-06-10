from moa.classifiers.trees import FIMTDD
from capymoa.base import MOARegressor
from capymoa.regressor import KNNRegressor
from capymoa.stream import stream_from_file
from capymoa.stream import CSVStream

#fried_stream = stream_from_file("../power_plant/test_data.csv", target="SlrW_Avg", drop=["TmStamp", "RecNum", "SlrMJ_Tot"])

fried_stream = CSVStream("../power_plant/test_data.csv", 
                        dtypes={
                            'Rain_mm':float,
                            'batt_volt': float,
                            'mean_wind_speed': float,
                            'mean_wind_direction': float,
                            'std_wind_dir': float,
                            'Max_Gust_Min': float,
                            'Max_Gust_Hr': float,
                            'Barametric_Avg': float,
                            'Air_Temp_Avg': float,
                            'RH_Avg': float,
                            'in_bytes_str': float,
                            'Dew_Point_Avg': float,
                            'SlrW_Avg': float,
                        },
                        class_index=12,
                        skip_header=True,
                        )

# fimtdd = MOARegressor(schema=fried_stream.get_schema(), moa_learner=FIMTDD())
# knnreg = KNNRegressor(schema=fried_stream.get_schema(), k=3, window_size=1000)

# results_fimtdd = prequential_evaluation(stream=fried_stream, learner=fimtdd, window_size=5000)
# results_knnreg = prequential_evaluation(stream=fried_stream, learner=knnreg, window_size=5000)

results_fimtdd['windowed'].metrics_per_window()
# Selecting the metric so that we don't use the default one.
# Note that the metric is different from the ylabel parameter, which just overrides the y-axis label.
plot_windowed_results(results_fimtdd, results_knnreg, metric="coefficient of determination")