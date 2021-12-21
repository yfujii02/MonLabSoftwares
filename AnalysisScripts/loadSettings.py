import yaml

def load_analysis_conf(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        AnalysisWindow = config[0]["AnalysisWindow"]
        Filtering      = config[1]["Filtering"]
        Histogram      = config[2]["Histogram"]
    return AnalysisWindow,Filtering,Histogram

#AnalysisWindow,Filtering,Histogram = load_analysis_conf("analysis_settings.yaml")
#print(AnalysisWindow["Start"])
#print(Filtering["MovingAveragePoints"])
#print(Histogram["NumberOfBins"])
