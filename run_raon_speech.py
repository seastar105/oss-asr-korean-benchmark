import json
import openai
import argparse
from jiwer import cer
from whisper_normalizer.basic import BasicTextNormalizer
from tqdm.auto import tqdm
import time
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import time


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--openai_api_base", type=str, required=True, help="Base URL for the VLLM Server.")
    parser.add_argument("--parallel_requests", type=int, default=32, help="Number of parallel requests to send.")
    
    return parser


def transcription_call(client, audio_path, model_name="KRAFTON/Raon-Speech-9B"):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
                {"type": "text", "text": "Transcribe the audio into text."}
            ]
        }
    ]
    cnt = 0
    while cnt < 5:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during API call for {audio_path}: {e}. Retrying...")
            time.sleep(2 ** cnt)  # Exponential backoff
            cnt += 1
    return ""  # Return empty string if all retries fail
    


def send_request(client, audio_path, model_name="KRAFTON/Raon-Speech-9B"):
    duration = librosa.get_duration(path=audio_path)
    start = time.time()
    transcription = transcription_call(client, audio_path, model_name=model_name)
    end = time.time()
    return transcription, duration, end - start


if __name__ == "__main__":
    args = get_parser().parse_args()

    client = openai.OpenAI(base_url=args.openai_api_base, api_key="EMPTY")

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    normalizer = BasicTextNormalizer()
    with ThreadPoolExecutor(max_workers=args.parallel_requests) as executor:
        future_to_item = {executor.submit(send_request, client, item["audio_path"], model_name="KRAFTON/Raon-Speech-9B"): item for item in data}
        for future in tqdm(as_completed(future_to_item), total=len(data), desc=f"Processing {args.input_jsonl}..."):
            item = future_to_item[future]
            try:
                transcription, duration, elapsed_time = future.result()
                ref = normalizer(item["text"])
                hyp = normalizer(transcription)
                results.append({
                    "key": item["key"],
                    "text": item["text"],
                    "audio_path": item["audio_path"],
                    "transcription": transcription,
                    "cer": cer(ref, hyp),
                    "duration": duration,
                    "elapsed_time": elapsed_time,
                })
            except Exception as e:
                print(f"Error processing {item['audio_path']}: {e}")

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            print(json.dumps(item, ensure_ascii=False), file=f)

    average_cer = sum(item["cer"] for item in results) / len(results)
    total_duration = sum(item["duration"] for item in results)
    total_elapsed_time = sum(item["elapsed_time"] for item in results)
    average_rtf = total_elapsed_time / total_duration if total_duration > 0 else float('inf')
    print(f"Processed {len(results)} audio files with {args.parallel_requests} parallel requests.")
    print(f"Average CER: {average_cer*100:.2f}%")
    print(f"Average RTF: {average_rtf:.4f}, Model processed {1/average_rtf:.2f} times faster than real-time., Total duration: {total_duration:.2f} seconds, Total elapsed time: {total_elapsed_time:.2f} seconds.")
