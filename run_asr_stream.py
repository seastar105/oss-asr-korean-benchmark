import json
import openai
import argparse
from jiwer import cer
from whisper_normalizer.basic import BasicTextNormalizer
from tqdm.auto import tqdm
import time
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import base64
import websockets
import asyncio


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="asr", help="Name of the ASR model on remote server.")
    parser.add_argument("--openai_api_base", type=str, required=True, help="Base URL for the VLLM Server.")
    parser.add_argument("--parallel_requests", type=int, default=32, help="Number of parallel requests to send.")
    
    return parser


async def realtime_call(client, audio_path, model_name="asr"):
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=f,
            model=model_name,
            language="ko",
            response_format="json",
            temperature=0.0
        )
    return response.text


def audio_to_pcm16_base64(audio_path: str) -> str:
    """
    Load an audio file and convert it to base64-encoded PCM16 @ 16kHz.
    """
    # Load audio and resample to 16kHz mono
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    # Convert to PCM16
    pcm16 = (audio * 32767).astype(np.int16)
    # Encode as base64
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")


async def realtime_transcribe(audio_path: str, host: str, port: int, model: str):
    """
    Connect to the Realtime API and transcribe an audio file.
    """
    uri = f"ws://{host}:{port}/v1/realtime"
    async with websockets.connect(uri) as ws:
        # Wait for session.created
        response = json.loads(await ws.recv())

        # Validate model
        await ws.send(json.dumps({"type": "session.update", "model": model}))

        # Signal ready to start
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Convert audio file to base64 PCM16
        audio_base64 = audio_to_pcm16_base64(audio_path)

        # Send audio in chunks (4KB of raw audio = ~8KB base64)
        chunk_size = 4096
        audio_bytes = base64.b64decode(audio_base64)
        total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size

        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )

        # Signal all audio is sent
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        while True:
            response = json.loads(await ws.recv())
            if response["type"] == "transcription.delta":
                pass
            elif response["type"] == "transcription.done":
                transcription = response["text"]
                return transcription
            elif response["type"] == "error":
                print(f"\nError: {response['error']}")
                break


def run_realtime_transcribe_sync(audio_path: str, host: str, port: int, model: str):
    """
    ThreadPoolExecutor에서 돌릴 수 있도록 async 함수를 sync wrapper로 감싼다.
    각 thread가 자기 event loop를 만들고 종료한다.
    """
    return asyncio.run(
        realtime_transcribe(
            audio_path=audio_path,
            host=host,
            port=port,
            model=model,
        )
    )


def send_request(host, port, audio_path, model_name="asr"):
    duration = librosa.get_duration(path=audio_path)
    start = time.time()
    transcription = run_realtime_transcribe_sync(audio_path, host, port, model_name)
    end = time.time()
    return transcription, duration, end - start


if __name__ == "__main__":
    args = get_parser().parse_args()

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    normalizer = BasicTextNormalizer()
    audio_path = data[0]["audio_path"]
    host = args.openai_api_base.replace("http://", "").split(":")[0]
    port = int(args.openai_api_base.split(":")[-1].split("/")[0])
    # print(asyncio.run(realtime_transcribe(audio_path, host=host, port=port, model=args.model_name)))
    
    
    with ThreadPoolExecutor(max_workers=args.parallel_requests) as executor:
        future_to_item = {executor.submit(send_request, host, port, item["audio_path"], model_name=args.model_name): item for item in data}
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
