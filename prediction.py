import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor
import argparse
from model.modeling_wav2vec import *
from utils.data_utils import *
from utils.dataloader import *
import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    config = AutoConfig.from_pretrained(args.saved_model)
    processor = Wav2Vec2Processor.from_pretrained(args.saved_model)
    sampling_rate = processor.feature_extractor.sampling_rate
    model = Wav2Vec2ForSpeechClassification.from_pretrained(args.saved_model).to(device)

    def speech_file_to_array_fn(path, sampling_rate):
        speech_array, _sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech


    def predict(path, sampling_rate):
        speech = speech_file_to_array_fn(path, sampling_rate)
        features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        input_values = features.input_values.to(device)
        attention_mask = features.attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
        return outputs

    STYLES = """
        <style>
        div.display_data {
            margin: 0 auto;
            max-width: 500px;
        }
        table.xxx {
            margin: 50px !important;
            float: right !important;
            clear: both !important;
        }
        table.xxx td {
            min-width: 300px !important;
            text-align: center !important;
        }
        </style>
        """.strip()
    def prediction(df_row):
        path, emotion = df_row["path"], df_row["emotion"]
        df = pd.DataFrame([{"Emotion": emotion, "Sentence": "    "}])
        setup = {
            'border': 2,
            'show_dimensions': True,
            'justify': 'center',
            'classes': 'xxx',
            'escape': False,
        }
        ipd.display(ipd.HTML(STYLES + df.to_html(**setup) + "<br />"))
        speech, sr = torchaudio.load(path)
        speech = speech[0].numpy().squeeze()
        speech = librosa.resample(np.asarray(speech), sr, sampling_rate)
        ipd.display(ipd.Audio(data=np.asarray(speech), autoplay=True, rate=sampling_rate))

        outputs = predict(path, sampling_rate)
        r = pd.DataFrame(outputs)
        ipd.display(ipd.HTML(STYLES + r.to_html(**setup) + "<br />"))
    
    test = pd.read_csv(args.val_dataset, sep="\t")
    print(prediction(test.iloc[0]))

if __name__ == "__main__": 
    main()
