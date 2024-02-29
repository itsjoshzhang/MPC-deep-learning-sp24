import os
import torch
import random
import basic_model as fwd
import deep_models as rnn
import matplotlib.pyplot as plt

SAMPLE = False
EPOCHS = 10000
device = torch.device("cpu")

def eval_basic(one_model = False):
    curr_h  = 0
    dataset = fwd.RobotData(fwd.DATA)

    for file in sorted(os.listdir("fwd_models")):
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

def eval_deep(one_model = False):
    curr_h  = 0
    curr_ft = 0
    dataset = rnn.dataset

    for file in sorted(os.listdir("rnn_models")):
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
        
        build_labels(labels[0])
        build_result(output[0])

        diff += torch.mean(torch.abs(output - labels)).item()
        rmse += torch.sqrt(torch.mean((output - labels) ** 2)).item()
    
    print(f"\n{file}")
    print(f"Avg. difference: {(diff / epochs):.4f}")
    print(f"Avg. rmsq error: {(rmse / epochs):.4f}")


labels = {"x_pos": [],
              "y_pos": [],
              "v_long": [],
              "v_tran": []}

def build_labels(label):
    x_pos = label[0]
    y_pos = label[1]
    # e = label[2]
    # w = label[3]
    v_long = label[4]
    v_tran = label[5]

    labels["x_pos"].append(x_pos)
    labels["y_pos"].append(y_pos)
    labels["v_long"].append(v_long)
    labels["v_tran"].append(v_tran)

result = {"x_pos": [],
            "y_pos": [],
            "v_long": [],
            "v_tran": []}

def build_result(output):
    x_pos = output[0]
    y_pos = output[1]
    # e = point[2]
    # w = point[3]
    v_long = output[4]
    v_tran = output[5]

    result["x_pos"].append(x_pos)
    result["y_pos"].append(y_pos)
    result["v_long"].append(v_long)
    result["v_tran"].append(v_tran)

def visualize_results():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    attrs = list(result.keys())
    axes = [ax1, ax2, ax3, ax4]
    
    for _, (attr, ax) in enumerate(zip(attrs, axes)):
        ax.plot(result[attr], label='prediction')
        ax.plot(labels[attr], label='actual')
        ax.set_title(attr)
        ax.legend()
    plt.show()

if __name__ == "__main__":
    """
    Evaluate all files in rnn
    w/ mean abs. error & rmse
    if SAMPLE = True:
        Use 10k random labels
    else: All 60k data labels
    """
    eval_basic(one_model = True)
    visualize_results()
    #eval_deep()