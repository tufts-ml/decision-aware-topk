# Decision Aware Maximum Likelihood
This anonymous repository contains the code for the paper "Decision-aware training of spatiotemporal forecasting models to select a top K subset of sites for intervention," currently in submission

## Synthetic Data Experiment
To replicate the results of the synthetic data experiment, run the following command:
```
python torch_frontier_experiment.py --step_size 0.1 \
    --perturbed_noise 0.01 \
    --epochs 2000 \
    --bpr_weight 30 --nll_weight 1 \
    --seed 360 \
    --outdir ./output/  \
    --num_components 2 \
    --init_idx 0 \
    --epsilon 0.84
```

Try adjusting epsilon to encourage the model to achieve different BPRs. The model will reach the target BPRs quickly, but the minimum NLL requires several thousand epochs.

## Opioid-related Overdose and Whooping Crane experiments
We share the publically available Cook County and Aerial Survellance datasets, preprocessed to fit our modeling strategy. Our experiments can be run with commands like the following:

```
python torch_opioid_exp.py --step_size 0.1 \
    --perturbed_noise 0.01 \
    --epochs 20 \
    --bpr_weight 30 \
    --nll_weight 1 \
    --seed 360 
    --data_dir ./data/cook_county/ \
    --device cuda \
    --outdir ./output/ \
    --epsilon 1
```

## Adding a new dataset
To add a new dataset, simply process your data into a long csv format, where a row corresponds to a single location and time. Both the x and y csvs should have columns called 'geoid' and 'timestep'. The labels should be in a column called 'counts'. Every other column in the x csv will be treated as a feature

## Adding a new model
To add a new model, define a new model class in torch_models.py, subclassing `torch.nn.Module`. The following methods are required:
    - forward
    - sample
    - log_likelihood
    - params_to_single_tensor
    - single_tensor_to_params
    - build_from_single_tensor
