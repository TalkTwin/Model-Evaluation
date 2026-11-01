import sys
sys.path.insert(0, "/home/aryannavale/projects")
from benchmark_worker import scan_inputs
for i, p in enumerate(scan_inputs("/home/aryannavale/projects/inputs"), 1):
    print(i, p["tag"], p["video"].split("/")[-1], p["audio"].split("/")[-1])
