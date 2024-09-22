from aria_glasses_utils.common import *
from aria_glasses_utils.BetterAriaProvider import *

from pathlib import Path
from pyzbar.pyzbar import decode
import os, glob, csv
import tomllib

config = tomllib.load(open("config.toml", "rb"))

def getSyncedVRSFiles(vrs_path):
    sessionFolder = Path(vrs_path).parents[1]
    return glob.glob(str(sessionFolder / "**/*.vrs"), recursive=True)

def save_info(file_path, start_timestamp, end_timestamp, scene, participant, take):
    csv_file = config["aria_recordings"]["recordings_sheet"]
    csv_file_exists = True

    if not os.path.exists(csv_file):
        csv_file_exists = False

    with open(csv_file, "a") as f:
        writer = csv.writer(f)

        if not csv_file_exists:
            # save header once
            fields = [
                "session_id",
                "file_paths",
                "start_timestamp",
                "end_timestamp",
                "scene",
                "participant",
                "take",
            ]
            writer.writerow(fields)

        fields = [
            Path(file_path).parts[-3],
            getSyncedVRSFiles(file_path),
            start_timestamp,
            end_timestamp,
            scene,
            participant,
            take,
        ]
        writer.writerow(fields)


def extract_codes(number):
    assert type(number) == int, "Number must be an integer."
    assert number < 1_000_000_000, "Number must be less than 1,000,000,000."

    scene = number // 1_000_000
    participant = (number % 1_000_000) // 1000
    take = number % 1000

    return scene, participant, take

END_CODE = 0

def decode_qrcode(img):
    result = decode(img)
    if len(result) > 0:
        num = int(result[0].data)
        if num == END_CODE:
            return "end", 0, 0, 0
        else:
            scene, participant, take = extract_codes(num)
            return "start", scene, participant, take

    return None, None, None, None


def main():

    folder_path = config["aria_recordings"]["vrs_glob"]

    for filename in glob.iglob(os.path.join(folder_path, "**/*.vrs"), recursive=True):
        file_path = os.path.join(folder_path, filename)
        print(f"Elaborating file {file_path}")

        provider = BetterAriaProvider(vrs=file_path)

        start_timestamp, end_timestamp, scene, participant, take = 0, 0, 0, 0, 0

        started = False

        for time in provider.get_time_range(time_step=1_000_000_000):
            print(f"INFO: Checking frame at time {time}")
            frame = {}

            frame["rgb"], _ = provider.get_frame(Streams.RGB, time_ns=time)

            result, res_scene, res_scene, res_take = decode_qrcode(frame["rgb"])

            if result == "start" and not started:
                started = True
                start_timestamp = time
                scene, participant, take = res_scene, res_scene, res_take
                print("INFO: ### DETECTED START QR-CODE")

            if result == "end" and start_timestamp != 0:
                end_timestamp = time
                print(
                    "INFO: ### DETECTED END QR-CODE",
                    file_path,
                    start_timestamp,
                    end_timestamp,
                    scene,
                    participant,
                    take,
                )
                save_info(
                    file_path, start_timestamp, end_timestamp, scene, participant, take
                )
                started = False
                start_timestamp = 0


if __name__ == "__main__":
    main()
