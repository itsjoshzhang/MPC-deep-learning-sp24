import pickle

"""
NAME:   (SIZE)
---------------
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

DATA = pickle.load(open("data_old.pkl", "rb"))

def print_keys():
    for data in DATA.items():
        print(f"\nData ({len(DATA)} items): Keys = {data[0]}, Vals = <dict> Bag.")

        for bag in data[1].items():
            print(f"Bag ({len(data[1])} items): Keys = {bag[0]}, Vals = <dict> Msg.")
            print(f"Msg ({len(bag[1])} items):  Keys =\n")

            for msg in bag[1].items():
                vals = msg[1]
                if isinstance(vals, dict):
                    vals = set(vals.keys())
                print(f"{msg[0]} = {vals}")
            return

def setup_data(CUT_TIME=1.00, BAG_SIZE=100, PAD=True):
    """
    Returns nested list DATA:
        data[x] = bag
        data[x][y] = msg
        data[x][y][z] = point
    Removes msgs within 1 sec
        of end of test/rosbag
    Limits max length of bags
        to 100, pruning extra
    """
    data_list = []
    for bag in DATA.values():
        bag_list = []
                                                # sort by incr. msg[time]
        bag_sort = sorted(bag.values(), key=lambda msg: msg["t"])
        max_time = bag_sort[-1]["t"] - CUT_TIME # max cutoff time for bag

        for msg in bag_sort:
            if msg["t"] <= max_time:
                bag_list.append([

                    msg["x"]["x"],          # 0. "x" (pos)   = {'x',
                    msg["x"]["y"],          # 1.                'y'}
                    msg["e"]["psi"],        # 2. "e" (orient)= {'psi'}
                    msg["w"]["w_psi"],      # 3. "w" (ang.v) = {'w_psi'}
                #   msg["aa"]["a_psi"],     # _. "aa"(ang.a) = {'a_psi'}
                    msg["v"]["v_long"],     # 4. "v" (veloc) = {'v_long',
                    msg["v"]["v_tran"],     # 5.                'v_tran'}
                #   msg["a"]["a_long"],     # _. "a" (accel) = {'a_long',
                #   msg["a"]["a_tran"],     # _.                'a_tran'}
                    msg["u"]["u_a"],        # 6."u" (actuat)= {'u_a',
                    msg["u"]["u_steer"]     # 7.               'u_steer'}
                ])
            if len(bag_list) == BAG_SIZE:   # Limit max length of bag
                data_list.append(bag_list)  # @ BAG_SIZE, prune extra
                bag_list = []

        tol = 8 if PAD else 0               # allow bags near 100-tol
        if len(bag_list) >= BAG_SIZE - tol: # pad to len 100 w/ zeros

            for _ in range(BAG_SIZE - len(bag_list)):
                bag_list.append([0 for _ in range(tol)])
            data_list.append(bag_list)

    size = lambda func: sum([func([len(msg) for msg in bag]) for bag in data_list])
    print(f"\n{len(data_list)} rosbags, {size(len)} messages, and {size(sum)} datapoints parsed.")
    return data_list

if __name__ == "__main__":
    data_old = setup_data(CUT_TIME=0.00, BAG_SIZE=1, PAD=False)
    data_new = setup_data(CUT_TIME=1.00, BAG_SIZE=100, PAD=True)
    # pickle.dump(data_new, open("data_new.pkl", "wb"))