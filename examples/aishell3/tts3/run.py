import subprocess

debug_mode = True
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
        preprocess()
    except:
        print("preprocess failed.")
        return


if __name__ == "__main__":
    do()
