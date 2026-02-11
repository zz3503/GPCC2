import os


def rename_sampling_folders(root_dir=r"."):
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name.startswith("sampling-"):
                old_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, "sampling")
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                else:
                    print(f"Skipped: {old_path} (Target name 'sampling' already exists)")


if __name__ == "__main__":
    rename_sampling_folders()
