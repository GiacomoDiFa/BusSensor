import json
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

# Configuration reader
def loadConfig(filename):
    with open(filename,"r") as file:
        return json.load(file)

# InfluxDB Manager
class InfluxManager:
    def __init__(self,config):
        self.cfg = config

        self.client = influxdb_client.InfluxDBClient(
            url = self.cfg['influx']['url'],
            org = self.cfg['influx']['org'],
            token = self.cfg['influx']['token']
        )

        self.apiWriter = self.client.write_api(write_options = SYNCHRONOUS)

    def sendCount(self, count: int):
        task = "monitor_task,host={},location={},room={}".format(self.cfg['influx']['host'],self.cfg['influx']['location'],self.cfg['influx']['room'])
        sequence = [str(task) + " bus-stop-count=" + str(count),]
        self.apiWriter.write(bucket = self.cfg['influx']['bucket'], org = self.cfg['influx']['org'], record = sequence)
    
    def sendDelta(self, delta: int):
        task = "monitor_task,host={},location={},room={}".format(self.cfg['influx']['host'],self.cfg['influx']['location'],self.cfg['influx']['room'])
        sequence = [str(task) + " bus-stop-delta=" + str(delta),]
        self.apiWriter.write(bucket = self.cfg['influx']['bucket'], org = self.cfg['influx']['org'], record = sequence)
