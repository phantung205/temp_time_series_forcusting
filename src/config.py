import os

# root path project
root_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#--------------------------------
#path data
#--------------------------------
data_dir = os.path.join(root_project,"data")
# path data raw
raw_data_path = os.path.join(data_dir,"raw","DailyDelhiClimateTrain.csv")
# dir data preprocessing
processed_data_dir = os.path.join(data_dir,"processed")


#--------------------------------
# path reports
#--------------------------------
report_dir = os.path.join(root_project,"reports")
# dir report eda
eda_report_dir = os.path.join(report_dir,"eda")
# dir report result
result_report_dir = os.path.join(report_dir,"result")
# dir results image
result_image = os.path.join(report_dir,"image_result")
#file name report
file_name_report = "repot_temperature.html"

#--------------------------------
# models path
#--------------------------------
model_dir = os.path.join(root_project,"models")

#--------------------------------
# test size and ramdom state
#--------------------------------
test_size = 0.2
random_state = 42

#--------------------------------
# target_size
#--------------------------------
ts = 3


#--------------------------------
# repuired columns
#--------------------------------
targets = ["target_{}".format(i+1) for i in range(ts)]
numerical_col = ["month","meantemp","meantemp_1","meantemp_2","meantemp_3","meantemp_4","humidity","wind_speed","meanpressure"]



