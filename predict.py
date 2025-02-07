import importlib
import os
import tempfile
import subprocess
import shutil
import time
from typing import List
from cog import BasePredictor, Input, Path

WEIGHTS_BASE_URL = "https://weights.replicate.delivery/default/yue/"


class Predictor(BasePredictor):
    def download_weights(self, filename: str, dest_dir: str):
        os.makedirs(dest_dir, exist_ok=True)

        if not os.path.exists(f"{dest_dir}/{filename}"):
            print(f"⏳ Downloading {filename} to {dest_dir}")
            start = time.time()
            subprocess.check_call(
                [
                    "pget",
                    "--log-level",
                    "warn",
                    "-xf",
                    f"{WEIGHTS_BASE_URL}/{filename}.tar",
                    dest_dir,
                ],
                close_fds=False,
            )
            print(f"✅ Download completed in {time.time() - start:.2f} seconds")
        else:
            print(f"✅ {filename} already exists in {dest_dir}")

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        cog_version = importlib.metadata.version("cog")
        print(f"Cog version: {cog_version}\n")
        models = [
            "models--m-a-p--YuE-s1-7B-anneal-en-cot",
            "models--Alissonerdx--YuE-s2-1B-general-int8",
            "models--Alissonerdx--YuE-s1-7B-anneal-en-cot-nf4",
            "models--Alissonerdx--YuE-s1-7B-anneal-en-cot-int8",
            "xcodec_mini_infer",
        ]

        for model in models:
            dest_dir = (
                "/src/inference/models" if "models--" in model else "/src/inference"
            )
            self.download_weights(model, dest_dir)

    def predict(
        self,
        genre_description: str = Input(
            description="Text containing genre tags that describe the musical style (e.g. instrumental, genre, mood, vocal timbre, vocal gender)",
            default="inspiring female uplifting pop airy vocal electronic bright vocal vocal",
        ),
        lyrics: str = Input(
            description="Lyrics for music generation. Must be structured in segments with [verse], [chorus], [bridge], or [outro] tags",
            default="[verse]\nOh yeah, oh yeah, oh yeah\n\n[chorus]\nOh yeah, oh yeah, oh yeah",
        ),
        num_segments: int = Input(
            description="Number of segments to generate", default=2, ge=1, le=10
        ),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to generate",
            default=1500,
            ge=500,
            le=3000,
        ),
        seed: int = Input(
            description="Set a seed for reproducibility. Random by default.",
            default=None,
        ),
        quantization_stage1: str = Input(
            description="Quantization stage 1",
            default="bf16",
            choices=["bf16", "int8", "nf4"],
        ),
        quantization_stage2: str = Input(
            description="Quantization stage 2",
            default="bf16",
            choices=["bf16", "int8"],
        ),
        use_dual_tracks_prompt: bool = Input(
            description="Enable dual-track ICL mode with separate vocal and instrumental tracks",
            default=False
        ),
        vocal_track_prompt: Path = Input(
            description="Path to the vocal track audio file for dual-track ICL mode",
            default=None
        ),
        instrumental_track_prompt: Path = Input(
            description="Path to the instrumental track audio file for dual-track ICL mode",
            default=None
        ),
        prompt_start_time: int = Input(
            description="Start time in seconds for the audio prompt",
            default=0,
            ge=0
        ),
        prompt_end_time: int = Input(
            description="End time in seconds for the audio prompt",
            default=30,
            ge=0,
            le=120
        ),
    ) -> List[Path]:
        """Run YuE inference on the provided inputs"""
        seed = self.seed_or_random_seed(seed)

        # Validate inputs
        if not lyrics.strip():
            raise ValueError("Lyrics cannot be empty")

        if not any(
            tag in lyrics.lower()
            for tag in ["[verse]", "[chorus]", "[bridge]", "[outro]"]
        ):
            raise ValueError(
                "Lyrics must contain at least one [verse], [chorus], [bridge], or [outro] tag"
            )

        if not genre_description.strip():
            raise ValueError("Genre description cannot be empty")

        # Validate dual-track mode inputs
        if use_dual_tracks_prompt:
            if not vocal_track_prompt or not instrumental_track_prompt:
                raise ValueError("Both vocal and instrumental track prompts are required for dual-track mode")
            if not os.path.exists(vocal_track_prompt):
                raise ValueError(f"Vocal track file does not exist: {vocal_track_prompt}")
            if not os.path.exists(instrumental_track_prompt):
                raise ValueError(f"Instrumental track file does not exist: {instrumental_track_prompt}")
            if prompt_end_time <= prompt_start_time:
                raise ValueError("prompt_end_time must be greater than prompt_start_time")

        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write inputs to files
            genre_file = os.path.join(temp_dir, "genre.txt")
            lyrics_file = os.path.join(temp_dir, "lyrics.txt")
            with open(genre_file, "w", encoding="utf-8") as f:
                f.write(genre_description)
            with open(lyrics_file, "w", encoding="utf-8") as f:
                f.write(lyrics)

            # Prepare command
            cmd = [
                "python",
                "/src/inference/infer.py",
                "--cuda_idx", "0",
                "--stage1_model", self.get_stage1_model(quantization_stage1),
                "--stage2_model", self.get_stage2_model(quantization_stage2),
                "--genre_txt", genre_file,
                "--lyrics_txt", lyrics_file,
                "--run_n_segments", str(num_segments),
                "--stage2_batch_size", "4",
                "--output_dir", temp_dir,
                "--max_new_tokens", str(max_new_tokens),
            ]

            # Add seed if specified
            if seed is not None:
                cmd.extend(["--seed", str(seed)])

            # Add dual-track mode parameters if enabled
            if use_dual_tracks_prompt:
                cmd.extend([
                    "--use_dual_tracks_prompt",
                    "--vocal_track_prompt_path", str(vocal_track_prompt),
                    "--instrumental_track_prompt_path", str(instrumental_track_prompt),
                    "--prompt_start_time", str(prompt_start_time),
                    "--prompt_end_time", str(prompt_end_time)
                ])

            # Run inference
            subprocess.run(cmd, check=True)

            # Find output files in vocoder/mix directory and rename to output_N.mp3
            mix_dir = os.path.join(temp_dir, "vocoder", "mix")
            output_files = []
            if os.path.exists(mix_dir):
                mp3_files = [f for f in os.listdir(mix_dir) if f.endswith(".mp3")]
                for idx, file in enumerate(mp3_files):
                    old_path = os.path.join(mix_dir, file)
                    new_name = (
                        "output.mp3" if len(mp3_files) == 1 else f"output_{idx+1}.mp3"
                    )
                    new_path = os.path.join(mix_dir, new_name)
                    os.rename(old_path, new_path)
                    output_files.append(Path(new_path))

            return output_files

    def seed_or_random_seed(self, seed: int | None) -> int:
        # Max seed is 2147483647
        if not seed or seed <= 0:
            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

        print(f"Using seed: {seed}\n")
        return seed

    def get_stage1_model(self, quantization_stage1: str) -> str:
        stage_1_model = {
            "bf16": "m-a-p/YuE-s1-7B-anneal-en-cot",
            "int8": "Alissonerdx/YuE-s1-7B-anneal-en-cot-int8",
            "nf4": "Alissonerdx/YuE-s1-7B-anneal-en-cot-nf4",
        }
        return stage_1_model[quantization_stage1]

    def get_stage2_model(self, quantization_stage2: str) -> str:
        stage_2_model = {
            "bf16": "m-a-p/YuE-s2-1B-general",
            "int8": "Alissonerdx/YuE-s2-1B-general-int8",
        }
        return stage_2_model[quantization_stage2]
