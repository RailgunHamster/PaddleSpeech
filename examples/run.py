import subprocess

debug_mode = True
manual_download = False
example1 = "aishell3"
example2 = "tts3"
model = "fastspeech2"
root = "C:/github/PaddleSpeech"
ckpt = "snapshot_iter_482.pdz"
example_path = f"{root}/examples/{example1}/{example2}"
temp = f"{root}/temp"
raw = f"{temp}/AISHELL-3"
aligned = f"{temp}/aligned"
trained = f"{temp}/trained"
dump = f"{temp}/dump"
duration = f"{temp}/durations.txt"
utils = f"{root}/utils"
tool = f"{root}/paddlespeech/t2s/exps/{model}"
config = f"{example_path}/conf/default.yaml"


def run(args: list[str]):
    args.insert(0, "python")
    state = subprocess.run(args)
    if debug_mode:
        if state.returncode:
            input(f"{args}\n failed with {state.returncode}, enter to continue.")
    else:
        state.check_returncode()
    return state


def download():
    if not manual_download:
        return
    import nltk
    import os
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
    print("Generate durations.txt from MFA results ...")
    run(
        [
            f"{utils}/gen_duration_from_textgrid.py",
            f"--inputdir={aligned}",
            f"--output",
            f"{duration}",
            f"--config={config}",
        ]
    )
    print("Extract features ...")
    run(
        [
            f"{tool}/preprocess.py",
            f"--dataset={example1}",
            f"--rootdir={raw}",
            f"--dumpdir={dump}",
            f"--dur-file={duration}",
            f"--config={config}",
            f"--num-cpu=20",
            f"--cut-sil=True",
        ],
    )
    print("Get features' stats ...")
    run(
        [
            f"{utils}/compute_statistics.py",
            f"--metadata={dump}/train/raw/metadata.json1",
            '--field-name="speech"',
        ],
    )
    run(
        [
            f"{utils}/compute_statistics.py",
            f"--metadata={dump}/train/raw/metadata.json1",
            '--field-name="pitch"',
        ],
    )
    run(
        [
            f"{utils}/compute_statistics.py",
            f"--metadata={dump}/train/raw/metadata.json1",
            '--field-name="energy"',
        ],
    )
    print("Normalize ...")
    run(
        [
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
        download()
        preprocess()
    except subprocess.CalledProcessError:
        print("preprocess failed.")
        return
    except Exception as e:
        print(e)
        return


if __name__ == "__main__":
    do()
