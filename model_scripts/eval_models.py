import os
import sys
import torch
import random
import basic_model as fwd
import deep_models as rnn
import matplotlib.pyplot as plt

SAMPLE = True
EPOCHS = 10000
device = torch.device("cpu")


def eval_basic(one_model = True):
    curr_h  = 0
    dataset = fwd.RobotData(fwd.DATA)

    for file in sorted(os.listdir("../model_data/fwd_models")):
        name = file.split("_")
        model_type = eval(f"fwd.{name[0]}_Model")
        
        inputs = fwd.INPUTS
        output = fwd.OUTPUT
        hidden = int(name[1])
        layers = int(name[2])

        model = model_type(inputs, hidden, output, layers, False).to(device)
        model.load_state_dict(torch.load(f"fwd_models/{file}", map_location = "cpu"))
        
        if curr_h != hidden:
            print("\n=======================")
            curr_h = hidden

        calc_error(dataset, model, file)
        if one_model: return

def eval_deep(one_model = True):
    curr_h  = 0
    curr_ft = 0
    dataset = rnn.dataset

    for file in sorted(os.listdir("../model_data/rnn_models")):
        name = file.split("_")
        model_type = eval(f"rnn.{name[0]}_Model")
        
        inputs = rnn.INPUTS
        output = rnn.OUTPUT
        hidden = int(name[1])
        layers = int(name[2])

        ft_size = int(name[3])
        state_f = eval(name[-1][:-3]) # -1 gets last word, -3 to avoid .pt

        model = model_type(inputs, hidden, output, layers, state_f).to(device)
        model.load_state_dict(torch.load(f"rnn_models/{file}", map_location = "cpu"))
        
        if curr_h != hidden:
            print("\n=======================")
            curr_h = hidden

        if curr_ft != ft_size:
            dataset = rnn.RobotData(rnn.DATA, ft_size)
        
        calc_error(dataset, model, file, state_f)
        if one_model: return

def calc_error(dataset, model, file, state_f = False):

    diff, rmse = 0, 0
    epochs = EPOCHS if SAMPLE else len(dataset)

    for i in range(epochs):
        if SAMPLE:
            i = random.randint(0, len(dataset) - 1)
        
        featrs, labels = dataset[i]
        featrs = featrs.to(device).unsqueeze(0)
        labels = labels.to(device).unsqueeze(0)

        batch_s = featrs.size(0)
        if state_f:
            model.init_hidden(batch_s)

        model.eval()
        with torch.no_grad():
            output = model(featrs)
        
        build_test(labels[0], params)
        build_test(output[0], result)

        diff += torch.mean(torch.abs(output - labels)).item()
        rmse += torch.sqrt(torch.mean((output - labels) ** 2)).item()
    
    print(f"\n{file}")
    print(f"Avg. difference: {(diff / epochs):.4f}")
    print(f"Avg. rmsq error: {(rmse / epochs):.4f}")


params = {"x_pos": [],
          "y_pos": [],
          "v_long": [],
          "v_tran": []}

result = {"x_pos": [],
          "y_pos": [],
          "v_long": [],
          "v_tran": []}

def build_test(in_arr, out_dict):
    x_pos = in_arr[0]
    y_pos = in_arr[1]
    # e = in_arr[2]
    # w = in_arr[3]
    v_long = in_arr[4]
    v_tran = in_arr[5]

    out_dict["x_pos"].append(x_pos)
    out_dict["y_pos"].append(y_pos)
    out_dict["v_long"].append(v_long)
    out_dict["v_tran"].append(v_tran)


def view_test():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    attrs = list(result.keys())
    axes = [ax1, ax2, ax3, ax4]
    
    for _, (attr, axis) in enumerate(zip(attrs, axes)):
        axis.plot(result[attr], label='prediction')
        axis.plot(params[attr], label='actual')
        axis.set_title(attr)
        axis.legend()
    plt.show()

if __name__ == "__main__":
    """
    Evaluate all files in rnn
    w/ mean abs. error & rmse
    if SAMPLE = True:
        Use 10k random labels
    else: All 60k data labels
    """
    args = sys.argv
    if len(args) != 3:
        raise SyntaxError("2 arguments required: (name of function), (True/False for SAMPLE)")
    
    method = eval(sys.argv[1])
    SAMPLE = eval(sys.argv[2])
    method(one_model = True)
    view_test()