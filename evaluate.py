from distutils.command.config import config

from tokenizers import Tokenizer
from model.modeling_wav2vec import *
from utils.data_utils import *
from utils.dataloader import *
import argparse
from utils.trainer import *
import librosa
from sklearn.metrics import classification_report

from transformers import TrainingArguments
def parsers(): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #dataset
    parser.add_argument("--val_dataset", default="./data/test.csv", type=str,
                        help="Path of the val dataset.")
    parser.add_argument("--saved_model", default="None", type=str,
                        help="Path of the val dataset.")

    args = parser.parse_args()
    return args

def main(): 
    args = parsers()
    test_dataset = load_dataset("csv", data_files={"test": args.val_dataset}, delimiter="\t")["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    config = AutoConfig.from_pretrained(args.saved_model)
    processor = Wav2Vec2Processor.from_pretrained(args.saved_model)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(args.saved_model).to(device)

    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        speech_array = speech_array.squeeze().numpy()
        speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)

        batch["speech"] = speech_array
        return batch


    def predict(batch):
        features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

        input_values = features.input_values.to(device)
        attention_mask = features.attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits 

        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        batch["predicted"] = pred_ids
        return batch
    
    test_dataset = test_dataset.map(speech_file_to_array_fn)
    result = test_dataset.map(predict, batched=True, batch_size=8)
    label_names = [config.id2label[i] for i in range(config.num_labels)]
    print("label_names: ", label_names)
    y_true = [config.label2id[name] for name in result["emotion"]]
    y_pred = result["predicted"]

    print(classification_report(y_true, y_pred, target_names=label_names))

if __name__ == "__main__":
    main()