import yaml


def load_dev(config_file):
    with open(config_file,"r") as f:
        config = yaml.safe_load(f)
        DeviceSettings  = config[0]["DeviceSettings"]
        DaqSettings     = config[1]["DaqSettings"]
        ChannelSettings = config[2]["ChannelSettings"]
    return DeviceSettings, DaqSettings, ChannelSettings
