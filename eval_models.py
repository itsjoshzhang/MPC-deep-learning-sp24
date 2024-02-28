import os
import torch
import random
import basic_model as fwd
import deep_models as rnn

def eval_basic(sample):
    """
    Evaluate all files in fwd
    w/ mean abs. error & rmse
    if sample = True:
        Use 10k random points
    else: All 60k data points
    """
    dataset = tm.RobotData(tm.DATA)
    curr_h  = 0

    for file in sorted(os.listdir(folder)):
        name = file.split("_")
        model_type = eval(f"tm.{name[0]}_Model")
        
        inputs = tm.INPUTS
        output = tm.OUTPUT
        hidden = int(name[1])
        layers = int(name[2])

        model = model_type(inputs, hidden, output, layers, False)
        model.load_state_dict(torch.load(f"{folder}/{file}", map_location="cpu"))
        
        if curr_h != hidden:
            print("\n=======================")
            curr_h = hidden

        diff, rmse = 0, 0
        if sample: epochs = 30000
        else: epochs = len(dataset)

        for i in range(epochs):
            if sample:
                i = random.randint(0, len(dataset) - 1)
            featrs, labels = dataset[i]
            
            featrs = featrs.to(tm.device).unsqueeze(0)
            labels = labels.to(tm.device).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                output = model(featrs)
            
            diff += torch.mean(torch.abs(output - labels)).item()
            rmse += torch.sqrt(torch.mean((output - labels) ** 2)).item()
        
        print(f"\n{file}")
        print(f"Avg. difference: {(diff / epochs):.4f}")
        print(f"Avg. rmsq error: {(rmse / epochs):.4f}")

def eval_basic(sample):
    """
    Evaluate all files in rnn
    w/ mean abs. error & rmse
    if sample = True:
        Use 10k random points
    else: All 60k data points
    """
    dataset = tm.RobotData(tm.DATA)
    curr_h  = 0

    for file in sorted(os.listdir(folder)):
        name = file.split("_")
        model_type = eval(f"tm.{name[0]}_Model")
        
        inputs = tm.INPUTS
        output = tm.OUTPUT
        hidden = int(name[1])
        layers = int(name[2])

        model = model_type(inputs, hidden, output, layers, False)
        model.load_state_dict(torch.load(f"{folder}/{file}", map_location="cpu"))
        
        if curr_h != hidden:
            print("\n=======================")
            curr_h = hidden

        diff, rmse = 0, 0
        if sample: epochs = 30000
        else: epochs = len(dataset)

        for i in range(epochs):
            if sample:
                i = random.randint(0, len(dataset) - 1)
            featrs, labels = dataset[i]
            
            featrs = featrs.to(tm.device).unsqueeze(0)
            labels = labels.to(tm.device).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                output = model(featrs)
            
            diff += torch.mean(torch.abs(output - labels)).item()
            rmse += torch.sqrt(torch.mean((output - labels) ** 2)).item()
        
        print(f"\n{file}")
        print(f"Avg. difference: {(diff / epochs):.4f}")
        print(f"Avg. rmsq error: {(rmse / epochs):.4f}")

if __name__ == "__main__":
    eval_files("fwd_models", sample = True) # See docstring for details