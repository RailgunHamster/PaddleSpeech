import os
import subprocess
import pyhocon

debug_mode = True
manual_download = False
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
    if debug_mode:
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
    if not manual_download:
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


def copy_exists():
    print("Copy phone_id_map.txt & speaker_id_map.txt & durations.txt")
    mission = [
        ["phone_set.txt", "phone_id_map.txt"],
        ["spk_info.txt", "speaker_id_map.txt"],
        ["train/label.txt", "durations.txt"],
    ]
    for m in mission:
        f = f"{data}/{m[0]}"
        t = f"{dump}/{m[1]}"
        with open(f, encoding="utf-8") as fc:
            lines = fc.readlines()
        with open(t, "w", encoding="utf-8") as fc:

            def filter_func(line: str):
                line = line.strip()
                return len(line) > 0 and not line.startswith("#")

            fc.writelines(filter(filter_func, lines))


def preprocess():
    if run_config.get("lexicon", False):
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
    if run_config.get("reorganize", False):
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
    if run_config.get("detect_oov", False):
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
    if run_config.get("mfa", False):
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
    if run_config.get("gen_duration", False):
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
    if run_config.get("preprocess", False):
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
    if run_config.get("compute_statistics", False):
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
    if run_config.get("normalize", False):
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


def do():
    try:
        create_folders()
        download()
        # copy_exists()
        preprocess()
    except subprocess.CalledProcessError:
        print("preprocess failed.")
        return
    except Exception as e:
        print(e)
        return


if __name__ == "__main__":
    do()
