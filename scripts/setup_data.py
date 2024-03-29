import pickle
import sys

"""
Rosbag pkl file format
DICT:   (SIZE)
=================
Data:    (80)
    Keys = "DAgger_XXX_GPS_0.db3"
    Vals (100+) = <dict> Bag
Bag:
    Key  (1)    = <int>  ID
    Val  (1)    = <dict> Msg
Msg:
    Keys (15):
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

def print_keys(pkl_data):
    """
    Print keys of rosbag msg.
    Use to debug data_raw.pkl
    """
    for data in pkl_data.items():
        print(f"\nData ({len(pkl_data)} items): Keys = {data[0]}, Vals = <dict> Bag.")

        for bag in data[1].items():
            print(f"Bag ({len(data[1])} items): Keys = {bag[0]}, Vals = <dict> Msg.")
            print(f"Msg ({len(bag[1])} items):  Keys =\n")

            for msg in bag[1].items():
                vals = msg[1]
                if isinstance(vals, dict):
                    vals = set(vals.keys())
                print(f"{msg[0]} = {vals}")
            return

def setup_data(pkl_data, cut_time = 1.00):
    """
    Returns nested list data:
        data[x] = bag
        data[x][y] = msg
        data[x][y][z] = point
    Removes msgs within 1 sec
        of end of test/rosbag
    """
    data_list = []
    for bag in pkl_data.values():
        bag_list = []
                                                # Sort by incr. msg[time]
        bag_sort = sorted(bag.values(), key = lambda msg: msg["t"])
        max_time = bag_sort[-1]["t"] - cut_time # Max cutoff time for bag

        for msg in bag_sort:
            if msg["t"] <= max_time:
                bag_list.append([

                    msg["x"]["x"],          # 0. "x" (pos)   = {'x',
                    msg["x"]["y"],          # 1.                'y'}
                    msg["e"]["psi"],        # 2. "e" (orient)= {'psi'}
                    msg["w"]["w_psi"],      # 3. "w" (ang.v) = {'w_psi'}
                    msg["v"]["v_long"],     # 4. "v" (veloc) = {'v_long',
                    msg["v"]["v_tran"],     # 5.                'v_tran'}
                    msg["u"]["u_a"],        # 6."u" (actuat)= {'u_a',
                    msg["u"]["u_steer"]     # 7.               'u_steer'}
                ])
        data_list.append(bag_list)

    size = lambda func: sum([func([len(msg) for msg in bag]) for bag in data_list])
    print(f"\n{len(data_list)} rosbags, {size(len)} messages, and {size(sum)} datapoints parsed.")
    return data_list

if __name__ == "__main__":
    args = sys.argv

    if len(args) != 2:
        raise SyntaxError("1 argument required: (name of pkl file)")

    pkl_file = sys.argv[1]
    pkl_data = pickle.load(open(pkl_file, "rb"))

    print_keys(pkl_data)
    data = setup_data(pkl_data)

    pkl_name = pkl_file.split(".")[0]
    pickle.dump(data, open(f"{pkl_name}_clean.pkl", "wb"))