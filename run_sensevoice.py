from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import argparse
import json
from jiwer import cer


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.")
    
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    model_dir = "iic/SenseVoiceSmall"
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        device="cuda:0",
    )
    
    audio_paths = [item["audio_path"] for item in data]
    
    res = model.generate(
        input=audio_paths,
        cache={},
        language="ko",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size=64,
    )
    preds = [rich_transcription_postprocess(item["text"]) for item in res]
    results = []
    for item, pred in zip(data, preds):
        results.append({
            "key": item["key"],
            "text": item["text"],
            "audio_path": item["audio_path"],
            "transcription": pred,
            "cer": cer(item["text"], pred),
        })
    
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            print(json.dumps(item, ensure_ascii=False), file=f)
    average_cer = sum(item["cer"] for item in results) / len(results)
    print(f"Average CER: {average_cer*100:.2f}%")