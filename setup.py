import pickle

"""
DATA:
    Keys = "DAgger_XXX_GPS_0.db3"
    Vals = <dict> L1
L1:
    Keys = <int> ID
    Vals = <dict> L2
L2:
    Keys:
    "header"    = {'stamp': {'sec': int, 'nanosec': int}, 'frame_id': str}
    "t" (time)  = <FLOAT>
    "x" (pos)   = {'header': ..., 'x': <FLOAT>, 'y': <FLOAT>, 'z': ...}
    "e" (orient)= {'phi': ..., 'theta': ...., 'psi': <FLOAT>}
    "q" (quatrn)= ...
    "w" (ang.v) = {'w_phi': ..., 'w_theta': ..., 'w_psi': <FLOAT>}
    "aa"(ang.a) = {'a_phi': ..., 'a_theta': ..., 'a_psi': <FLOAT>}
    "v" (veloc) = {'v_long': <FLOAT>, 'v_tran': <FLOAT>, 'v_n': ...}
    "a" (accel) = {'a_long': <FLOAT>, 'a_tran': <FLOAT>, 'a_n': ...}
    "p" (par.p) = ...
    "pt" (par.v)= ...
    "u" (actuat)= {'t': ..., 'u_a': <FLOAT>, 'u_steer': <FLOAT>, 'timing': ...}
    "hw" (drive)= ...
    "timing"    = {'step_start': ..., 'step_exec': ..., 'source': ..., 'publish': ...}
    "lap_num"   = ...
"""
DATA = pickle.load(open("data.pkl", "rb"))

def print_keys():
    for data in DATA.items():
        print(f"{sum([len(i) for i in DATA.values()])} messages seen.")
        print(f"Data ({len(DATA)} items): Keys = {data[0]}, Vals = <dict> L1.")

        for L1 in data[1].items():
            print(f"L1 ({len(data[1])} items): Keys = {L1[0]}, Vals = <dict> L2.")
            print(f"\nL2 ({len(L1[1])} items): Keys =")

            for L2 in L1[1].items():
                vals = L2[1]
                if isinstance(vals, dict):
                    vals = set(vals.keys())
                print(f"{L2[0]} = {vals}")
            exit()

def parse_data():
    """
    Returns DATA as a nested list:
    list    = DATA rosbag dirs(80)
    list[x] = L1 data bundles (>1)
    list[x][y] = L2 data bundle(1)
    list[x][y][z] = datapoint (12)
    """
    list = []
    for L1 in DATA.values():
        nest = []
        for L2 in L1.values():
            nest.append([
                L2["t"],            # 0. "t" (time)  = <FLOAT>
                L2["x"]["x"],       # 1. "x" (pos)   = {'x',
                L2["x"]["y"],       # 2.                'y'}
                L2["e"]["psi"],     # 3. "e" (orient)= {'psi'}
                L2["w"]["w_psi"],   # 4. "w" (ang.v) = {'w_psi'}
                L2["aa"]["a_psi"],  # 5. "aa"(ang.a) = {'a_psi'}
                L2["v"]["v_long"],  # 6. "v" (veloc) = {'v_long',
                L2["v"]["v_tran"],  # 7.                'v_tran'}
                L2["a"]["a_long"],  # 8. "a" (accel) = {'a_long',
                L2["a"]["a_tran"],  # 9.                'a_tran'}
                L2["u"]["u_a"],     # 10."u" (actuat)= {'u_a',
                L2["u"]["u_steer"]  # 11.               'u_steer'}
            ])
        list.append(nest)

    size = lambda func: sum([func([len(i) for i in nest]) for nest in list])
    print(f"\n{len(list)} rosbags, {size(len)} messages, and {size(sum)} datapoints parsed.")
    return list

def prune_data():
    return

if __name__ == "__main__":
    x = parse_data()
    print_keys()