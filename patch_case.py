import yaml

with open("cases/cicular_vent/case.yaml", "r") as f:
    case = yaml.safe_load(f)

case["components"][0]["geometry"]["type"] = "thick_cylinder"
case["components"][0]["geometry"]["parameters"] = {
    "radius_inner": 0.060,
    "radius_outer": 0.065,
    "length": 1.0,
    "center": [0.0, 0.0, 0.0],
    "axis": "z"
}

with open("cases/cicular_vent/case.yaml", "w") as f:
    yaml.dump(case, f, sort_keys=False)

