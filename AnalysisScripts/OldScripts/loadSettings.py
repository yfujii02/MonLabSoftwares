import yaml

def load_analysis_conf(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        Waveform       = config[0]["Waveform"]
        AnalysisWindow = config[1]["AnalysisWindow"]
        Filtering      = config[2]["Filtering"]
        Histogram      = config[3]["Histogram"]
    return Waveform,AnalysisWindow,Filtering,Histogram

#AnalysisWindow,Filtering,Histogram = load_analysis_conf("analysis_settings.yaml")
#print(AnalysisWindow["Start"])
#print(Filtering["MovingAveragePoints"])
#print(Histogram["NumberOfBins"])
