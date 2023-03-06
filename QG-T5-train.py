import os, wandb, argparse, yaml
from transformers import T5Tokenizer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from utils_dataset import *
from utils_model import LightningT5Module

os.environ['WANDB_NOTEBOOK_NAME'] = 'multimodal-QG-train'
seed_everything(21, workers=True)

    
def add_params():
    parser = argparse.ArgumentParser()
    # Data IO
    parser.add_argument("--dataset_dir", type=str, 
                        default="/data/zw16/dataset_ScienceQA", 
                        help="Path to the data directory")
    parser.add_argument("--data_keep_mode", type=int, default=1,
                        help="How to keep the data. 1: hint+image, 2: image+lecture, 3: image+lecture+hint, 4: image+lecture+NO hint")
    parser.add_argument("--descriptions_file", type=str,
                        default='generated_descriptions/blip2-flan-t5-xxl_halfprecTrue_prompt0_gmodeC_pa0.6_topk4_temp1_topp0.95_spTrue_nsp20_minl30_maxl100_seed42.csv', 
                        help="Path to the descriptions file")
    parser.add_argument("--extracted_texts_file", type=str,
                        default='extracted_texts_img/extracted_texts.json',
                        help="Path to the extracted texts file")
    parser.add_argument("--desc_sel_mode", type=int, default=2,
                        help="How to select the description. 1: random, 2: ppl rerank")
    parser.add_argument("--input_format_opt", type=int, default=1,
                        help="How to format the input. 1: hint+image context, 2: hint only, 3: image only, 4: lecture+image, 5: lecture only")
    parser.add_argument("--target_format_opt", type=int, default=1,
                        help="How to format the target. 1: question only; 2: question+choices; 3: hint+question")
    # Model configs
    parser.add_argument("-MN", "--model_name", type=str, default="t5-small", 
                        help="Variant of the Transformer model for finetuning")
    # Train configs
    parser.add_argument("-B", "--batch_size", type=int, default=3, 
                        help="Batch size for training the Transformer Model")
    parser.add_argument("-L", "--learning_rate", type=float, default=3e-4, 
                        help="Learning Rate for training the Transformer Model")
    parser.add_argument("-E", "--num_epochs", type=int, default=8, 
                        help="Total Number of Epochs")
    parser.add_argument("-ACC", "--accumulate_grad_batches", type=int, default=4, 
                        help="Num of batches to accumulate gradients for")
    parser.add_argument("-CLIP", "--gradient_clip_val", type=float, default=1.0, 
                        help="Gradient clipping value")
    parser.add_argument('-LC', '--load_checkpoint', action='store_true', 
                        help='Load Checkpoint for re-finetuning')
    parser.add_argument("-CN", "--checkpoint_name", type=str, default="flan_t5_large_codex_0.00_augment", 
                        help="Variant of the trained Transformer Base Model")
    parser.add_argument('-LP', '--linear_probing', action='store_true', 
                        help='For Linear Probing (Train only the lm head)')
    # Efficiency configs
    parser.add_argument('-TS', '--training_strategy', type=str, default="DP", 
                        help="DP for data parellel and DS for deepspeed")
    parser.add_argument("-D", "--num_devices", type=int, default=1, 
                        help="Devices used for training")
    parser.add_argument("-PRE", "--precision", type=int, default=32, 
                        help="Precision for training")
    parser.add_argument('-DG', '--debug', action='store_true', 
                        help='For Debugging')
    params = parser.parse_args()
    return params


if __name__ == '__main__':
    args = add_params()
    
    # Create run_name with model_name, model name extracted from descriptions_file, 
    # desc_sel_mode, input_format_opt, target_format_opt, 
    # batch_size, learning_rate, num_epochs, accumulate_grad_batches, gradient_clip_val
    run_name = f'{"-".join(args.model_name.split("/"))}_DataKeep{args.data_keep_mode}_descSel{args.desc_sel_mode}_inpFormat{args.input_format_opt}_tarFormat{args.target_format_opt}_bs{args.batch_size}_lr{args.learning_rate}_ep{args.num_epochs}_gradAcc{args.accumulate_grad_batches}_gradClip{args.gradient_clip_val}_descFile{args.descriptions_file.split("/")[-1].replace(".csv", "")}'

    problems, pid_splits, descriptions, extracted_texts = load_and_filter_raw_data(
        dataset_dir=args.dataset_dir, descriptions_file=args.descriptions_file, 
        extracted_texts_file=args.extracted_texts_file, data_keep_mode=args.data_keep_mode)
    inputs_train, targets_train, pids_train = format_io_data(
        problems, pid_splits, descriptions, extracted_texts, split='train', 
        desc_sel_mode=args.desc_sel_mode, input_format_opt=args.input_format_opt, 
        target_format_opt=args.target_format_opt)
    inputs_valid, targets_valid, pids_valid = format_io_data(
        problems, pid_splits, descriptions, extracted_texts, split='val', 
        desc_sel_mode=args.desc_sel_mode, input_format_opt=args.input_format_opt, 
        target_format_opt=args.target_format_opt)
    print('Loaded Data!')
    
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    print('Loaded T5 tokenizer!')
    
    src_len = 1024 if args.input_format_opt in set([4,5]) else 512 # use 1024 length if context is in input
    tgt_len = 512 if args.target_format_opt == 3 else 128 # use 512 length if hint is in target
    train_input_ids, train_attention_mask, train_labels = get_transformer_encoding(
        tokenizer, inputs_train, targets_train, src_len=src_len, tgt_len=tgt_len)
    valid_input_ids, valid_attention_mask, valid_labels = get_transformer_encoding(
        tokenizer, inputs_valid, targets_valid, src_len=src_len, tgt_len=tgt_len)
    print('Tokenized Data!')

    train_dataset = QGDataset(train_input_ids, train_attention_mask, train_labels)
    valid_dataset = QGDataset(valid_input_ids, valid_attention_mask, valid_labels)
    print('Created Pytorch Dataset')

    batch_size = args.batch_size
    training_dataloader = get_dataloader(batch_size, train_dataset)
    valid_dataloader = get_dataloader(batch_size, valid_dataset, datatype='val')
    print('Loaded Dataloader!')

    max_epochs = args.num_epochs

    # Load checkpoint or create new model from_pretrained
    if args.load_checkpoint:
        search_dir = os.path.join('./Checkpoints', args.checkpoint_name)
        for file in os.listdir(search_dir):
            ckpt_file = os.path.join(search_dir, file)
        print('ckpt_file', ckpt_file)
        model = LightningT5Module.load_from_checkpoint(ckpt_file)
        print('Successfully loaded the saved checkpoint!')
        save_name = 'reft_' + run_name
    else:
        model = LightningT5Module(
            model_name = args.model_name, lp=args.linear_probing, 
            training_dl=training_dataloader, valid_dl=valid_dataloader,
            num_train_epochs=max_epochs, lr=args.learning_rate)
        save_name = run_name

    if args.linear_probing:
        save_name = 'lp_' + save_name
            
    print('Save name:', save_name)

    # Logging
    wandb.login()
    logger = WandbLogger(name=save_name, project='multimodal-QG-train')
    
    # Monitering and call backs
    lr_monitor = LearningRateMonitor(logging_interval='step')    
    save_directory = os.path.join('./checkpoints', save_name)
    # save_checkpoint =  ModelCheckpoint(dirpath=save_directory, save_last=True)
    save_checkpoint =  ModelCheckpoint(
        dirpath=save_directory, monitor='validation_loss', save_top_k=1)
    early_stop_callback = EarlyStopping(
        monitor='validation_loss', patience=3, 
        strict=False, verbose=False, mode='min')

    # create dir if not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save a copy of the args as yaml file in the save directory
    with open(os.path.join(save_directory, 'args.yaml'), 'w') as f:
        yaml.dump(args, f)
    
    # Training strategy
    if args.training_strategy == 'DP':
        strategy = DDPStrategy(find_unused_parameters=False)
    elif args.training_strategy == 'DS':
        strategy = DeepSpeedStrategy(
            stage = 2, offload_optimizer=True,
            allgather_bucket_size=5e8, reduce_bucket_size=5e8)

    # Init trainer
    trainer = Trainer(accelerator='gpu', devices=args.num_devices, 
                    default_root_dir=save_directory, 
                    val_check_interval=0.5,
                    logger=logger,
                    max_epochs=max_epochs,
                    callbacks=[lr_monitor, save_checkpoint, early_stop_callback],
                    deterministic=True,
                    strategy = strategy,
                    accumulate_grad_batches=args.accumulate_grad_batches,
                    gradient_clip_val=args.gradient_clip_val,
                    precision=args.precision)

    # Training
    trainer.fit(model)

    print('Model Training Complete!')
    print('Saving model in path: {:s}'.format(save_directory))