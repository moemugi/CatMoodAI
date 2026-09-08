import os
import librosa

DATASET_PATH = "../CAT_DB"


def analyze_dataset():

    total_files = 0

    print("=" * 70)
    print("CAT SOUND DATASET ANALYSIS")
    print("=" * 70)

    classes = sorted(
        folder
        for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(
            os.path.join(DATASET_PATH, folder)
        )
    )

    print(f"\nNumber of classes: {len(classes)}")

    print("\nClasses:")
    for i, class_name in enumerate(classes):
        print(f"{i}: {class_name}")

    print("\n" + "=" * 70)
    print("RECORDINGS AND DURATIONS")
    print("=" * 70)

    for class_name in classes:

        folder_path = os.path.join(
            DATASET_PATH,
            class_name
        )

        audio_files = [
            file
            for file in os.listdir(folder_path)
            if file.lower().endswith(".mp3")
        ]

        durations = []

        for file in audio_files:

            path = os.path.join(
                folder_path,
                file
            )

            try:

                duration = librosa.get_duration(
                    path=path
                )

                durations.append(duration)

            except Exception as e:

                print(
                    f"\nCould not read: {path}"
                )

                print(e)

        if durations:

            average = sum(durations) / len(durations)
            shortest = min(durations)
            longest = max(durations)

            print(f"\n{class_name}")
            print(f"  Recordings : {len(audio_files)}")
            print(f"  Shortest   : {shortest:.2f} sec")
            print(f"  Longest    : {longest:.2f} sec")
            print(f"  Average    : {average:.2f} sec")

            total_files += len(audio_files)

    print("\n" + "=" * 70)
    print(f"TOTAL RECORDINGS: {total_files}")
    print("=" * 70)


if __name__ == "__main__":
    analyze_dataset()