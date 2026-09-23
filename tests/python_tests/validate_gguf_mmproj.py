# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GGUF multimodal acceptance against a pinned llama.cpp CPU oracle.

Build gguf_mmproj_oracle.cpp against REFERENCE_REVISION. No llama.cpp production dependency.
"""
import argparse
import json
import hashlib
import traceback
import subprocess  # nosec B404
import tempfile
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as genai
from PIL import Image

REFERENCE_REVISION = "03fa73cb27f5c251b9528489b18d303b1366aca4"


class Tokens(genai.StreamerBase):
    def __init__(self):
        super().__init__()
        self.tokens = []

    def write(self, tokens):
        self.tokens.extend(tokens if isinstance(tokens, list) else [tokens])
        return genai.StreamingStatus.RUNNING

    def end(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", type=Path)
    parser.add_argument("mmproj", type=Path)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--reference-language", type=Path, help="Optional F32 copy of represented weights for reference inference")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, help="Source manifest for an immutable runtime snapshot")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video-frames", type=int, default=0, help="Also compare a short synthetic video (Qwen/Gemma)")
    parser.add_argument("--audio", type=Path, help="Also compare audio and mixed image/audio requests (16 kHz mono WAV)")
    parser.add_argument("--api-checks", action="store_true", help="Check beam search, modern chat, streaming cancellation and reset")
    parser.add_argument("--multi-media", action="store_true", help="Compare two images and, when available, two audio inputs")
    parser.add_argument("--audio-boundaries", action="store_true", help="Compare audio immediately before and after the 30-second chunk boundary")
    parser.add_argument("--chat", action="store_true", help="Also compare a cached image chat follow-up")
    parser.add_argument("--family", choices=("gemma3", "gemma4", "qwen35", "muse"), default="gemma3")
    parser.add_argument("--attention-backend", choices=("SDPA", "PA"), default="SDPA")
    args = parser.parse_args()
    report = {"language": str(args.language), "mmproj": str(args.mmproj),
              "reference_revision": REFERENCE_REVISION, "family": args.family,
              "reference_language": str(args.reference_language or args.language),
              "audio": str(args.audio) if args.audio else None,
              "multi_media": args.multi_media, "audio_boundaries": args.audio_boundaries,
              "attention_backend": args.attention_backend,
              "openvino_version": ov.get_version(), "genai_version": genai.__version__,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": [], "chat_checked": args.chat, "passed": False, "completed": False}
    if args.runtime_manifest:
        report["runtime_sources"] = json.loads(args.runtime_manifest.read_text())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args, report)
        report["completed"] = True
        report["passed"] = (report["request_reset_matches"] and report["chat_reset_matches"] and
                            all(c["passed"] for c in report.get("api_checks", {}).values()) and
                            all(c["first_token_matches"] and c["matching_choice_fraction"] >= .9
                                for c in report["cases"]))
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


def run(args, report):
    image_marker = {"gemma3": "<start_of_image>", "gemma4": "<|image|>", "muse": "<|image|>",
                    "qwen35": "<|vision_start|><|image_pad|><|vision_end|>"}[args.family]
    image_prompt = image_marker + "\nDescribe the image."
    args.report.parent.mkdir(parents=True, exist_ok=True)
    pipe = genai.VLMPipeline(str(args.language), "CPU", mmproj_path=str(args.mmproj),
                            ATTENTION_BACKEND=args.attention_backend,
                            INFERENCE_PRECISION_HINT="f32", DYNAMIC_QUANTIZATION_GROUP_SIZE=0,
                            INFERENCE_NUM_THREADS=4)
    tokenizer = pipe.get_tokenizer()
    if args.image:
        pixels = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        pixels = np.zeros((64, 64, 3), np.uint8)
        pixels[..., 0] = 255
    cases = report["cases"]
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        image_file = directory / "image.png"
        Image.fromarray(pixels).save(image_file)

        def compare(messages, tokens, text, modality, with_image, with_audio=False, media_override=None, merge_frames=False):
            """Replay `tokens` through the oracle on the same history and score the agreement."""
            # GenAI expands image markers into embeddings. mtmd uses its own marker and adds
            # the same Gemma3 begin/end-image tokens around the reference encoder output.
            (directory / "prompt.txt").write_text(
                tokenizer.apply_chat_template(messages, add_generation_prompt=True))
            (directory / "history.txt").write_text(" ".join(map(str, tokens)))
            media = ([str(image_file)] if with_image else []) + ([str(args.audio.resolve())] if with_audio else [])
            if media_override is not None:
                media = media_override
            process = subprocess.run([str(args.oracle.resolve()), str((args.reference_language or args.language).resolve()),
                str(args.mmproj.resolve()), ";".join(media) if media else "-",
                str(directory / "prompt.txt"), str(directory / "history.txt")] + (["--merge-frames"] if merge_frames else []), capture_output=True, text=True)
            (args.report.parent / f"{args.report.stem}-{modality}.log").write_text(process.stderr)
            process.check_returncode()
            choices = next(line for line in process.stdout.splitlines() if line.startswith("CHOICES"))
            reference = list(map(int, choices.split()[1:]))
            assert len(reference) == len(tokens) and reference
            case = {"modality": modality, "text": text, "tokens": tokens,
                    "reference_choices_on_same_history": reference,
                    "first_token_matches": reference[0] == tokens[0],
                    "matching_choice_fraction": sum(a == b for a, b in zip(reference, tokens)) / len(reference)}
            cases.append(case)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(case), flush=True)
            return case

        for image in (False, True):
            prompt = image_prompt if image else "What is 2 plus 2?"
            stream = Tokens()
            kwargs = {"images": [ov.Tensor(pixels[None])]} if image else {}
            result = pipe.generate(prompt, max_new_tokens=20, do_sample=False, streamer=stream, **kwargs)
            compare([{"role": "user", "content": prompt.replace(image_marker, "<__media__>")}],
                    stream.tokens, result.texts[0], "image" if image else "text", image)
        if args.video_frames:
            assert args.family in ("qwen35", "gemma4"), "Add the family's reference video token assembly first"
            frames = np.stack([np.roll(pixels, i * 8, axis=1) for i in range(args.video_frames)])
            metadata = genai.VideoMetadata()
            metadata.fps = 2.
            metadata.frames_indices = list(range(args.video_frames))
            files = []
            for i, frame in enumerate(frames):
                file = directory / f"frame{i}.png"
                Image.fromarray(frame).save(file)
                files.append(str(file))
            if args.family == "qwen35":
                marker = "<|vision_start|><|video_pad|><|vision_end|>"
                reference_prompt = "".join(
                    f"<{(i + min(i + 1, len(frames) - 1)) / 4:.1f} seconds>" +
                    "<__media__>" * min(2, len(frames) - i) for i in range(0, len(frames), 2))
            else:
                marker = "<|video|>"
                reference_prompt = " ".join(f"00:{i // 2:02d} <__media__>" for i in range(len(frames)))
            stream = Tokens()
            result = pipe.generate(marker + "\nDescribe the video.", videos=[ov.Tensor(frames)],
                                   videos_metadata=[metadata], max_new_tokens=20, do_sample=False, streamer=stream)
            compare([{"role": "user", "content": reference_prompt + "\nDescribe the video."}],
                    stream.tokens, result.texts[0], "video", False, media_override=files,
                    merge_frames=args.family == "qwen35")
        if args.audio:
            import wave
            with wave.open(str(args.audio)) as audio_file:
                assert audio_file.getframerate() == 16000 and audio_file.getnchannels() == 1
                assert audio_file.getsampwidth() == 2
                waveform = np.frombuffer(audio_file.readframes(audio_file.getnframes()), dtype="<i2").astype(np.float32) / 32768
            for mixed in (False, True):
                prompt = (image_marker + "\n" if mixed else "") + "<|audio|>\nTranscribe the audio."
                kwargs = {"audios": [ov.Tensor(waveform)]}
                if mixed:
                    kwargs["images"] = [ov.Tensor(pixels[None])]
                stream = Tokens()
                result = pipe.generate(prompt, max_new_tokens=20, do_sample=False, streamer=stream, **kwargs)
                reference_prompt = prompt.replace(image_marker, "<__media__>").replace("<|audio|>", "<__media__>")
                compare([{"role": "user", "content": reference_prompt}], stream.tokens,
                        result.texts[0], "mixed" if mixed else "audio", mixed, True)
        if args.multi_media:
            second_pixels = np.ascontiguousarray(pixels.transpose(1, 0, 2))
            second_file = directory / "image2.png"
            Image.fromarray(second_pixels).save(second_file)
            prompt = image_marker + "\n" + image_marker + "\nCompare these images."
            stream = Tokens()
            result = pipe.generate(prompt, images=[ov.Tensor(pixels[None]), ov.Tensor(second_pixels[None])],
                                   max_new_tokens=20, do_sample=False, streamer=stream)
            compare([{"role": "user", "content": prompt.replace(image_marker, "<__media__>")}],
                    stream.tokens, result.texts[0], "multi_image", False,
                    media_override=[str(image_file), str(second_file)])
        if args.audio and (args.multi_media or args.audio_boundaries):
            def audio_file(samples, name):
                file = directory / (name + ".wav")
                with wave.open(str(file), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes((samples * 32768).astype("<i2").tobytes())
                return str(file)

            extra_audios = []
            if args.multi_media:
                extra_audios.append(("multi_audio", [waveform, np.ascontiguousarray(waveform[::-1])]))
            if args.audio_boundaries:
                for length in (30 * 16000 - 1, 30 * 16000 + 321):
                    extra_audios.append((f"audio_samples_{length}", [np.resize(waveform, length)]))
            for name, samples in extra_audios:
                prompt = "<|audio|>\n" * len(samples) + "Transcribe the audio."
                stream = Tokens()
                result = pipe.generate(prompt, audios=[ov.Tensor(sample) for sample in samples],
                                       max_new_tokens=20, do_sample=False, streamer=stream)
                files = [audio_file(sample, f"{name}_{i}") for i, sample in enumerate(samples)]
                compare([{"role": "user", "content": prompt.replace("<|audio|>", "<__media__>")}],
                        stream.tokens, result.texts[0], name, False, media_override=files)
        chat_reset_matches = True
        if args.chat:
            pipe.start_chat()
            first = Tokens()
            first_result = pipe.generate(image_prompt,
                images=[ov.Tensor(pixels[None])], max_new_tokens=20, do_sample=False, streamer=first)
            chat_reset_matches = first.tokens == cases[1]["tokens"]
            report["chat_initial_tokens"] = first.tokens
            followup = Tokens()
            followup_prompt = "What is shown?"
            followup_result = pipe.generate(followup_prompt, max_new_tokens=20,
                                            do_sample=False, streamer=followup)
            compare([{"role": "user", "content": "<__media__>\nDescribe the image."},
                     {"role": "assistant", "content": first_result.texts[0]},
                     {"role": "user", "content": followup_prompt}],
                    followup.tokens, followup_result.texts[0], "image_chat", True)
            pipe.finish_chat()
        if args.audio and args.chat:
            pipe.start_chat()
            first = Tokens()
            audio_prompt = "<|audio|>\nTranscribe the audio."
            first_result = pipe.generate(audio_prompt, audios=[ov.Tensor(waveform)],
                                         max_new_tokens=20, do_sample=False, streamer=first)
            followup = Tokens()
            followup_prompt = "What did the speaker say?"
            followup_result = pipe.generate(followup_prompt, max_new_tokens=20,
                                            do_sample=False, streamer=followup)
            compare([{"role": "user", "content": "<__media__>\nTranscribe the audio."},
                     {"role": "assistant", "content": first_result.texts[0]},
                     {"role": "user", "content": followup_prompt}],
                    followup.tokens, followup_result.texts[0], "audio_chat", False, True)
            pipe.finish_chat()
        if args.api_checks:
            checks = report["api_checks"] = {}

            def check(name, function):
                try:
                    function()
                    checks[name] = {"passed": True}
                except Exception:
                    checks[name] = {"passed": False, "error": traceback.format_exc()}
                finally:
                    pipe.finish_chat()
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                print(name, checks[name], flush=True)

            def modern_chat(audio=False):
                prompt = "<|audio|>\nTranscribe the audio." if audio else image_prompt
                media = {"audios": [ov.Tensor(waveform)]} if audio else {"images": [ov.Tensor(pixels[None])]}
                history = genai.ChatHistory([{"role": "user", "content": prompt}])
                stream = Tokens()
                result = pipe.generate(history, max_new_tokens=20, do_sample=False, streamer=stream, **media)
                reference_prompt = "<__media__>\nTranscribe the audio." if audio else "<__media__>\nDescribe the image."
                reference_history = [{"role": "user", "content": reference_prompt}]
                case = compare(reference_history, stream.tokens, result.texts[0],
                               "modern_audio" if audio else "modern_image", not audio, audio)
                first_case = case
                # Switch to a different history with identical media-token geometry.
                # Token IDs alone cannot distinguish the two encoders' outputs.
                other = genai.ChatHistory([{"role": "user", "content": prompt}])
                other_media = ({"audios": [ov.Tensor(np.ascontiguousarray(waveform[::-1]))]} if audio else
                               {"images": [ov.Tensor((255 - pixels)[None])]})
                pipe.generate(other, max_new_tokens=1, do_sample=False, **other_media)
                history.append({"role": "assistant", "content": result.texts[0]})
                reference_history.append({"role": "assistant", "content": result.texts[0]})
                followup = "What did the speaker say?" if audio else "What is shown?"
                history.append({"role": "user", "content": followup})
                reference_history.append({"role": "user", "content": followup})
                stream = Tokens()
                result = pipe.generate(history, max_new_tokens=20, do_sample=False, streamer=stream)
                case = compare(reference_history, stream.tokens, result.texts[0],
                               "modern_audio_chat" if audio else "modern_image_chat", not audio, audio)
                assert all(c["first_token_matches"] and c["matching_choice_fraction"] >= .9
                           for c in (first_case, case)), (first_case, case)

            def beam_search():
                result = pipe.generate(image_prompt, images=[ov.Tensor(pixels[None])],
                    num_beams=3, num_return_sequences=2, max_new_tokens=8, min_new_tokens=8, do_sample=False)
                assert len(result.texts) == 2 and np.isfinite(result.scores).all()

            def cancel_and_reset():
                class Cancel(Tokens):
                    def write(self, tokens):
                        super().write(tokens)
                        return genai.StreamingStatus.CANCEL if len(self.tokens) >= 3 else genai.StreamingStatus.RUNNING
                pipe.generate(image_prompt, images=[ov.Tensor(pixels[None])], max_new_tokens=20, do_sample=False,
                              streamer=Cancel())
                stream = Tokens()
                pipe.generate("What is 2 plus 2?", max_new_tokens=20, do_sample=False, streamer=stream)
                assert stream.tokens == cases[0]["tokens"], stream.tokens

            if args.chat:
                check("modern_image_chat", modern_chat)
                if args.audio:
                    check("modern_audio_chat", lambda: modern_chat(True))
            check("beam_search", beam_search)
            check("cancel_and_reset", cancel_and_reset)

        # A second request must start with an empty cache.
        reset_stream = Tokens()
        pipe.generate("What is 2 plus 2?", max_new_tokens=20, do_sample=False, streamer=reset_stream)
        reset_matches = reset_stream.tokens == cases[0]["tokens"]
    report.update(request_reset_matches=reset_matches, chat_reset_matches=chat_reset_matches)



if __name__ == "__main__":
    raise SystemExit(main())
