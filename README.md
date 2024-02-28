# File Structure

`eval_models.py`
- Evaluates all files in `fwd_models/` and `rnn_models/`
- Calculates difference in model output vs dataset label
- Can use 10,000 random labels or all 60,000 data labels

`basic_model.py`
- Train a basic linear feed-forward model for regression
- Moderately trained models can be used in `fwd_models/`
- **16 / 64 neurons, 2 layers, 32 batch size** work best

`deep_models.py`
- Trains RNNs (GRU and LSTM) using stateful or stateless
- Moderately trained models can be used in `rnn_models/`
- **16 / 64 neurons, ? layers, ? length, ? batch** work best