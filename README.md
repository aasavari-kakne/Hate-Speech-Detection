# Hate Speech Detection using Transfer Learning with RoBERTa

## Design choice 

### Data set creation
1. maximum sequence length = 200
2. Train split = 80% i.e. 16k examples
3. Test split = 10% i.e. 2k examples
4. Val split = 10% i.e. 2k examples


### Architecture 
1. Dropout probability = 0.1


### Training 
1. Optimizer = AdamW with learning rate = 2e-5
2. Batch size = 32
3. num_epochs = 20
4. class_weights = [1.0, 2.663]
