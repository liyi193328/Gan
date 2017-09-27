# Gan

This project mainly involves using gan to fill in missing data

## Train model
python train.py --model_class=autoencoder|gan --train_datapath=xx --model_name=aa

checkpoints will save to ./checkpoints/{model_name}/

logs will save to ./logs/{model_names}/


## Predict
python predict.py --model_class=autoencoder|gan --model_name=aa --checkpoint_dir=a --infer_complete_datapath=y --outDir=data/

predict result will save to ./{outDir}/{model_name}.infer.complete


