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
    "aa"(ang.aa)= {'a_phi': ..., 'a_theta': ..., 'a_psi': <FLOAT>}
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
        print(f"\nData ({len(DATA)} items): Keys = {data[0]}, Vals = <dict> L1")

        for L1 in data[1].items():
            print(f"\nL1 ({len(data[1])} items): Keys = {L1[0]}, Vals = <dict> L2")
            print(f"\nL2 ({len(L1[1])} items): Keys =")

            for L2 in L1[1].items():
                vals = L2[1]
                if isinstance(vals, dict):
                    vals = set(vals.keys())
                print(f"{L2[0]} = {vals}")
            exit()

def parse_data():
    return

def prune_data():
    return

if __name__ == "__main__":
    print_keys()