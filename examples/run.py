import os
import subprocess
import pyhocon

example1 = "aishell3"
example2 = "tts3"
model = "fastspeech2"
lexicon = "simple"
root = "C:/github/PaddleSpeech"
ckpt = "snapshot_iter_482.pdz"
example_path = f"{root}/examples/{example1}/{example2}"
temp = f"{root}/temp"
data = f"{temp}/AISHELL-3"
aligned = f"{temp}/aligned"
trained = f"{temp}/trained"
dump = f"{temp}/dump"
duration = f"{dump}/durations.txt"
utils = f"{root}/utils"
tool = f"{root}/paddlespeech/t2s/exps/{model}"
config = f"{example_path}/conf/default.yaml"
textgrids = f"{aligned}/aishell3_alignment_tone"

run_config = pyhocon.ConfigFactory.parse_file(f"{root}/examples/config.conf")


def run(args: list[str], condition: bool = True):
    if not condition:
        print("skip...")
        return
    state = subprocess.run(args)
    if run_config.get("debug_mode", False):
        if state.returncode:
            input(f"{args}\n failed with {state.returncode}, enter to continue.")
    else:
        state.check_returncode()
    return state


def create_folders():
    print("create output folders.")
    folders = [aligned, trained, dump]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def download():
    if not run_config.get("manual_download", False):
        print("auto downloads.")
        return
    print("manual downloads.")
    import nltk
    import sys

    # path gen from nltk/data.py file
    path = []
    _paths_from_env = os.environ.get("NLTK_DATA", "").split(os.pathsep)
    path += [d for d in _paths_from_env if d]
    if "APPENGINE_RUNTIME" not in os.environ and os.path.expanduser("~/") != "~/":
        path.append(os.path.expanduser("~/nltk_data"))
    path += [
        os.path.join(sys.prefix, "nltk_data"),
        os.path.join(sys.prefix, "share", "nltk_data"),
        os.path.join(sys.prefix, "lib", "nltk_data"),
        os.path.join(os.environ.get("APPDATA", "C:\\"), "nltk_data"),
        r"C:\nltk_data",
        r"D:\nltk_data",
        r"E:\nltk_data",
    ]
    ids = [
        {"id": "averaged_perceptron_tagger", "dir": "taggers", "ext": ".zip"},
        {"id": "cmudict", "dir": "corpora", "ext": ".zip"},
    ]
    try:
        for id in ids:
            for p in path:
                f = os.path.join(p, id["dir"], f"{id['id']}{id['ext']}")
                if os.path.exists(f):
                    break
            else:
                nltk.download(id["id"])
    except Exception as e:
        path_str = "".join("\n    - %r" % d for d in path)
        print(f'download nltk("{ids}") yourself into: {path_str}\n  or use proxy.')
        raise e


def preprocess():
    c = run_config.get("preprocess")
    if c is None:
        print("no preprocess config")
        return
    if c.get("lexicon", False):
        print("generating lexicon...")
        run(
            [
                "python",
                f"{mfa_local}/generate_lexicon.py",
                f"{aligned}/{lexicon}",
                "--with-r",
                "--with-tone",
            ],
            not os.path.exists(f"{aligned}/{lexicon}.lexicon"),
        )
        print("lexicon done")
    else:
        print("skip lexicon")
    if c.get("reorganize", False):
        print("reorganizing baker corpus...")
        run(
            [
                "python",
                f"{mfa_local}/reorganize_baker.py",
                f"--root-dir={raw}",
                f"--output-dir={aligned}/baker_corpus",
                "--resample-audio",
            ],
            not os.path.exists(f"{aligned}/baker_corpus"),
        )
        print(f"reorganization done. Check output in {aligned}/baker_corpus.")
        print("audio files are resampled to 16kHz")
        print(
            f"transcription for each audio file is saved with the same namd in {aligned}/baker_corpus "
        )
    else:
        print("skip reorganize")
    if c.get("detect_oov", False):
        print("detecting oov...")
        run(
            [
                "python",
                f"{mfa_local}/detect_oov.py",
                f"{aligned}/baker_corpus",
                f"{aligned}/{lexicon}.lexicon",
            ],
        )
        print(
            "detecting oov done. you may consider regenerate lexicon if there is unexpected OOVs."
        )
    else:
        print("skip detect_oov")
    if c.get("mfa", False):
        print("Start MFA training...")
        run(
            [
                "mfa",
                f"{aligned}/baker_corpus",
                f"{aligned}/{lexicon}.lexicon",
                f"{aligned}/baker_alignment",
                "-o",
                f"{aligned}/baker_model",
                "--clean",
                "--verbose",
                "-j",
                "10",
                "--temp_directory",
                f"{aligned}/.mfa_train_and_align",
            ],
            not os.path.exists(f"{aligned}/baker_alignment"),
        )
        print("training done!")
        print(f"results: {aligned}/baker_alignment")
        print(f"model: {aligned}/baker_model")
    else:
        print("skip mfa")
    if c.get("gen_duration", False):
        print("Generate durations.txt from MFA results ...")
        run(
            [
                "python",
                f"{utils}/gen_duration_from_textgrid.py",
                f"--inputdir={textgrids}",
                f"--output",
                f"{duration}",
                f"--config={config}",
            ],
        )
    else:
        print("skip gen_duration")
    if c.get("preprocess", False):
        print("Extract features ...")
        run(
            [
                "python",
                f"{tool}/preprocess.py",
                f"--dataset={example1}",
                f"--rootdir={data}",
                f"--dumpdir={dump}",
                f"--dur-file={duration}",
                f"--config={config}",
                f"--num-cpu=20",
                f"--cut-sil=True",
            ]
        )
    else:
        print("skip preprocess")
    if c.get("compute_statistics", False):
        print("Get features' stats ...")
        print("compute speech...")
        run(
            [
                "python",
                f"{utils}/compute_statistics.py",
                f"--metadata={dump}/train/raw/metadata.jsonl",
                "--field-name=speech",
            ],
        )
        print("compute pitch...")
        run(
            [
                "python",
                f"{utils}/compute_statistics.py",
                f"--metadata={dump}/train/raw/metadata.jsonl",
                "--field-name=pitch",
            ],
        )
        print("compute energy...")
        run(
            [
                "python",
                f"{utils}/compute_statistics.py",
                f"--metadata={dump}/train/raw/metadata.jsonl",
                "--field-name=energy",
            ],
        )
    else:
        print("skip compute_statistics")
    if c.get("normalize", False):
        print("Normalize ...")
        run(
            [
                "python",
                f"{tool}/normalize.py",
                f"--metadata={dump}/train/raw/metadata.jsonl",
                f"--dumpdir={dump}/train/norm",
                f"--speech-stats={dump}/train/speech_stats.npy",
                f"--pitch-stats={dump}/train/pitch_stats.npy",
                f"--energy-stats={dump}/train/energy_stats.npy",
                f"--phones-dict={dump}/phone_id_map.txt",
                f"--speaker-dict={dump}/speaker_id_map.txt",
            ],
        )
        run(
            [
                "python",
                f"{tool}/normalize.py",
                f"--metadata={dump}/dev/raw/metadata.jsonl",
                f"--dumpdir={dump}/dev/norm",
                f"--speech-stats={dump}/train/speech_stats.npy",
                f"--pitch-stats={dump}/train/pitch_stats.npy",
                f"--energy-stats={dump}/train/energy_stats.npy",
                f"--phones-dict={dump}/phone_id_map.txt",
                f"--speaker-dict={dump}/speaker_id_map.txt",
            ],
        )
        run(
            [
                "python",
                f"{tool}/normalize.py",
                f"--metadata={dump}/test/raw/metadata.jsonl",
                f"--dumpdir={dump}/test/norm",
                f"--speech-stats={dump}/train/speech_stats.npy",
                f"--pitch-stats={dump}/train/pitch_stats.npy",
                f"--energy-stats={dump}/train/energy_stats.npy",
                f"--phones-dict={dump}/phone_id_map.txt",
                f"--speaker-dict={dump}/speaker_id_map.txt",
            ],
        )
    else:
        print("skip normalize")


def train():
    c = run_config.get("train")
    if c is None:
        print("no train config")
        return
    """
    python ${BIN_DIR}/train.py \
        --train-metadata=${DUMP_DIR}/train/norm/metadata.jsonl \
        --dev-metadata=${DUMP_DIR}/dev/norm/metadata.jsonl \
        --config=${config_path} \
        --output-dir=${train_output_path} \
        --ngpu=2 \
        --phones-dict=${DUMP_DIR}/phone_id_map.txt \
        --speaker-dict=${DUMP_DIR}/speaker_id_map.txt
    """


def do():
    try:
        create_folders()
        download()
        preprocess()
        train()
    except subprocess.CalledProcessError:
        print("preprocess failed.")
        return
    except Exception as e:
        print(e)
        return


if __name__ == "__main__":
    do()
