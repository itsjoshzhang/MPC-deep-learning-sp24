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
DATA = pickle.load(open("data.pkl", "rb"))


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


def parse_data():
    """
    Returns nested list DATA:
        data[x] = bag
        data[x][y] = msg
        data[x][y][z] = point
    """
    data = []
    for B in DATA.values():
        bag = []
        for M in B.values():
            bag.append([
                M["t"],             # 0. "t" (time)  = <FLOAT>
                M["x"]["x"],        # 1. "x" (pos)   = {'x',
                M["x"]["y"],        # 2.                'y'}
                M["e"]["psi"],      # 3. "e" (orient)= {'psi'}
                M["w"]["w_psi"],    # 4. "w" (ang.v) = {'w_psi'}
            #   M["aa"]["a_psi"],   # _. "aa"(ang.a) = {'a_psi'}
                M["v"]["v_long"],   # 5. "v" (veloc) = {'v_long',
                M["v"]["v_tran"],   # 6.                'v_tran'}
            #   M["a"]["a_long"],   # _. "a" (accel) = {'a_long',
            #   M["a"]["a_tran"],   # _.                'a_tran'}
                M["u"]["u_a"],      # 7."u" (actuat)= {'u_a',
                M["u"]["u_steer"]   # 8.               'u_steer'}
            ])
        data.append(sorted(bag, key = lambda msg: msg[0]))

    size = lambda func: sum([func([len(msg) for msg in bag]) for bag in data])
    print(f"\n{len(data)} rosbags, {size(len)} messages, and {size(sum)} datapoints parsed.")
    return data


def prune_data(data, stops=True, ends=True):
    """
    Returns pruned data list:
    if stops: removes points
        if all(w,v,u) < 0.01
    if ends: removes points
        within 1s of msg end
    Flattens data list to 2D
    """
    # Adjust bounds as needed
    near_0 = lambda msg: all([abs(pt) < 0.01 for pt in msg])

    prune = []
    for bag in data:
        max_t = max([msg[0] for msg in bag]) - 1.00

        for msg in bag:
            if stops and near_0(msg[4:]): # (w,v,u)
                continue
            if ends and msg[0] > max_t:
                continue
            prune.append(msg)

    print(f"\n{len(prune)} messages and {sum([len(msg) for msg in prune])} datapoints remain from pruning.")
    return prune


if __name__ == "__main__":
    """
    Original: messages    datapoints
                64048       576432
    w/ stops:   -998        -8982   (1.6)%
    w/ ends:    -4041       -36369  (6.3)%
    w/ both:    -5032       -45288  (7.9)%
    """
    if DATA:
        print_keys()

        data = parse_data()
        pickle.dump(data, open("data_clean.pkl", "wb"))

        data = prune_data(data)
        pickle.dump(data, open("data_prune.pkl", "wb"))