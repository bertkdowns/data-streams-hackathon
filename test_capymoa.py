from capymoa.regressor import FIMTDD
from capymoa.base import MOARegressor
from capymoa.regressor import KNNRegressor, AdaptiveRandomForestRegressor
from capymoa.stream import stream_from_file
from capymoa.stream import CSVStream, stream_from_file
from capymoa.stream._stream import NumpyStream
from capymoa.evaluation import prequential_evaluation
from capymoa.evaluation.visualization import plot_windowed_results
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display
from pprint import pprint  # Import pprint module

#fried_stream = stream_from_file("../power_plant/test_data.csv", target="SlrW_Avg", drop=["TmStamp", "RecNum", "SlrMJ_Tot"])

# fried_stream = CSVStream("../power_plant/test_data.csv", 
#                         dtypes={
#                             'Rain_mm':float,
#                             'batt_volt': float,
#                             'mean_wind_speed': float,
#                             'mean_wind_direction': float,
#                             'std_wind_dir': float,
#                             'Max_Gust_Min': float,
#                             'Max_Gust_Hr': float,
#                             'Barametric_Avg': float,
#                             'Air_Temp_Avg': float,
#                             'RH_Avg': float,
#                             'in_bytes_str': float,
#                             'Dew_Point_Avg': float,
#                             'SlrW_Avg': float,
#                         },
#                         class_index=12,
#                         skip_header=True,
#                         )

path_to_csv_or_arff = Path("test_data.csv")
class_index: int = 12,
target_type: str = "numeric"  # "numeric" or "categorical"

x_features = pd.read_csv(path_to_csv_or_arff).to_numpy()
#np.genfromtxt(path_to_csv_or_arff, delimiter=",", skip_header=1)

print(x_features[0])
print(x_features.shape)
print(np.shape(x_features))
# columns: TmStamp,RecNum,batt_volt,mean_wind_speed,mean_wind_direction,std_wind_dir,Max_Gust_Min,Max_Gust_Hr,
# Rain_mm,Barametric_Avg,Air_Temp_Avg,RH_Avg,SlrW_Avg,SlrMJ_Tot,in_bytes_str,Dew_Point_Avg

# target is SlrW_Avg
targets = x_features[:, class_index]

# remove slrw_avg, slrmj_tot, recnum, tmstamp from x_features

x_features = np.delete(x_features, -2, axis=1) # remove in_bytes_str
x_features = np.delete(x_features, class_index, axis=1) # remove slrw_avg
x_features = np.delete(x_features, class_index, axis=1) # remove slrmj_tot
x_features = np.delete(x_features, 0, axis=1) # remove recnum
x_features = np.delete(x_features, 0, axis=1) # remove tmstamp


fried_stream = NumpyStream(
    x_features,
    targets,
    target_type=target_type,
    feature_names=["batt_volt", "mean_wind_speed", "mean_wind_direction", "std_wind_dir", "Max_Gust_Min", "Max_Gust_Hr","Rain_mm","Barametric_Avg","Air_Temp_Avg","RH_Avg","Dew_Point_Avg"],
)


ARF_learner = AdaptiveRandomForestRegressor(schema=fried_stream.get_schema(), ensemble_size=10)
knnreg = KNNRegressor(schema=fried_stream.get_schema(), k=3, window_size=1000)

results_arf = prequential_evaluation(stream=fried_stream, learner=ARF_learner, window_size=5000)
results_knnreg = prequential_evaluation(stream=fried_stream, learner=knnreg, window_size=5000)

pprint(results_arf['windowed'].metrics_per_window())
# Selecting the metric so that we don't use the default one.
# Note that the metric is different from the ylabel parameter, which just overrides the y-axis label.


pprint(results_arf)

#plot_windowed_results(results_arf, results_knnreg, metric="coefficient of determination")


#print(f'[ClassificationWindowedEvaluator] Windowed accuracy reported for every window_size windows')
#display(results_arf.metrics_per_window()[['classified instances','classifications correct (percent)']])