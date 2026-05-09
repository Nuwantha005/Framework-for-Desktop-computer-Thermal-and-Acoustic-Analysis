import yaml

with open("cases/cicular_vent/case.yaml", "r") as f:
    case = yaml.safe_load(f)

# Set DP to 0 or remove the actuator disk
case["actuator_disks"] = []
case["inlets"][0]["n_theta"] = 48
case["outlets"][0]["n_theta"] = 48

with open("cases/cicular_vent/case.yaml", "w") as f:
    yaml.dump(case, f, sort_keys=False)

