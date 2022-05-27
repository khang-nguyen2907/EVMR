from transformers import AutoConfig, Wav2Vec2Processor
from datasets import load_dataset, load_metric
import torchaudio

class DataLoader:
    def __init__(self, args) -> None:
        self.args = args
        self.input_column = "path"
        self.output_column = "emotion"
        self.model_name = self.args.model_name
        self.pooling_mode = self.args.pooling_mode
        self.train_dataset = self.get_dataset(self.args.train_dataset)
        self.eval_dataset = self.get_dataset(self.args.val_dataset)
        self.num_labels, self.label_list = self.get_classification_labels(self.train_dataset)

        self.config = AutoConfig.from_pretrained(
            self.model_name, 
            num_labels = self.num_labels, 
            label2id={label: i for i, label in enumerate(self.label_list)},
            id2label={i: label for i, label in enumerate(self.label_list)},
            finetuning_task="wav2vec2_clf",
        )
        setattr(self.config, 'pooling_mode', self.pooling_mode)

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
    
    def get_config(self): 
        return self.config
    
    def get_processor(self): 
        return self.processor

    def get_dataset(self): 
        data_files = {
            "train": self.args.train_dataset, 
            "validation": self.args.val_dataset
        }

        dataset = load_dataset("csv", data_files = data_files, delimiter="\t")
        train_dataset = dataset["train"]
        eval_dataset = dataset['validation']

        return train_dataset, eval_dataset

    def get_classification_labels(self, dataset): 
        

        label_list = dataset.unique(self.output_column)
        label_list.sort()
        num_labels = len(label_list)

        return num_labels, label_list
    
    def speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech
    
    def label_to_id(self, label, label_list):

        if len(label_list) > 0:
            return label_list.index(label) if label in label_list else -1

        return label
    
    def preprocess_function(self, examples):
        speech_list = [self.speech_file_to_array_fn(path) for path in examples[self.input_column]]
        target_list = [self.label_to_id(label, self.label_list) for label in examples[self.output_column]]

        result = self.processor(speech_list, sampling_rate=self.target_sampling_rate)
        result["labels"] = list(target_list)

        return result
    
    def get_train_val_dataset(self): 
        train_dataset = self.train_dataset.map(
            self.preprocess_function, 
            batch_size = self.args.batch_size, 
            batched = True, 
            num_proc = self.args.num_proc
        )

        eval_dataset = self.eval_dataset.map(
            self.preprocess_function, 
            batch_size = self.args.batch_size, 
            batched = True, 
            num_proc = self.args.num_proc
        )

        return train_dataset, eval_dataset
