from distutils.command.config import config

from tokenizers import Tokenizer
from model.modeling_wav2vec import *
from utils.data_utils import *
from utils.dataloader import *
import argparse
from utils.trainer import *

from transformers import TrainingArguments
def parsers(): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #dataset
    parser.add_argument("--train_dataset", default="./data/train.csv", type=str,
                        help="Path of the train dataset.")
    parser.add_argument("--val_dataset", default="./data/test.csv", type=str,
                        help="Path of the val dataset.")
    parser.add_argument("--saved_model", default="None", type=str,
                        help="Path of the val dataset.")
            

    #Model
    parser.add_argument("--batch_size", default=4, type=int,
                        help="batch size")
    parser.add_argument("--num_proc", default=4, type=int,
                        help="number of processes")
    parser.add_argument("--model_name", default="bert", type=str,
                        help="name of pretrained model.")
    parser.add_argument("--pooling_mode", default="mean", type=str,
                        help="['mean', 'sum', 'max']")
    
    #Training Argument
    parser.add_argument("--output_dir", default="./save_model", type=str,
                        help="Path of the folder holding saved model, other information.")
    parser.add_argument("--log_dir", default="./logs", type=str,
                        help="Path of the log folder ")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="gradient_accumulation_steps")
    parser.add_argument("--epoch_nums", default=5, type=int,
                        help="number of training epochs")
    parser.add_argument("--fp16", default=True, type=bool,
                        help="use fp16 or not")
    parser.add_argument("--save_steps", default=2, type=int,
                        help="save steps")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="learning_rate")

    args = parser.parse_args()
    return args
def main(): 
    args = parsers()

    #dataset
    data_loader = DataLoader(args=args)
    train_dataset, eval_dataset = data_loader.get_train_val_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #initialize model 
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        args.model_name, 
        config = data_loader.get_config()
    )
    if args.saved_model != "None":
        print (f'loading and initializing model from {args.saved_model}')
        model_state_dict, old_args = torch.load(args.load_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_state_dict)

    model.freeze_feature_extractor()
    model.to(device)

    #Training Arguments
    training_args = TrainingArguments(
        output_dir = args.output_dir, 
        loggibf_dir = args.log_dir,
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        num_train_epochs=args.epoch_nums,
        fp16=True,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        logging_steps=args.save_steps,
        learning_rate=args.learning_rate,
        save_total_limit=2,
    )

    processor = data_loader.get_processor()
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding = True)

    trainer = Torontorainer(
        model=model, 
        data_collator = data_collator, 
        args = training_args, 
        compute_metrics = compute_metrics, 
        train_dataset = train_dataset, 
        eval_dataset = eval_dataset, 
        tokenizer = processor.feature_extractor,
    )

if __name__ == "__main__": 
    main()