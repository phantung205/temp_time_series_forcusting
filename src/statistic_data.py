import pandas as pd
from src import config
import os
from ydata_profiling import ProfileReport


def generate_classifier_report():
    report_dir = config.eda_report_dir
    file_name = config.file_name_report

    # check directory
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    # read data raw
    df = pd.read_csv(config.raw_data_path)

    #create report
    profile = ProfileReport(df,title=file_name,explorative=True)

    # full directory path
    report_path = os.path.join(report_dir,file_name)

    # overwrite if the file already exists
    profile.to_file(report_path)

    print(f"reports create at: {report_path}")


if __name__ == '__main__':
    generate_classifier_report()