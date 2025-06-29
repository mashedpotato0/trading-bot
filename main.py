import os
import random
import shutil

# CONFIG
DATA_FOLDER = "./stock_data"
TRAIN_FOLDER = os.path.join(DATA_FOLDER, "train")
TEST_FOLDER = os.path.join(DATA_FOLDER, "test")
TEST_RATIO = 0.1  # 10% test, 90% train

# Ensure output folders exist
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# Get all .csv files in the base folder
all_csvs = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

# Shuffle randomly
random.shuffle(all_csvs)

# Split into train/test
split_index = int(len(all_csvs) * (1 - TEST_RATIO))
train_files = all_csvs[:split_index]
test_files = all_csvs[split_index:]

# Move files
def move_files(file_list, dest_folder):
    for f in file_list:
        src = os.path.join(DATA_FOLDER, f)
        dst = os.path.join(dest_folder, f)
        shutil.move(src, dst)

move_files(train_files, TRAIN_FOLDER)
move_files(test_files, TEST_FOLDER)

print(f"✅ Moved {len(train_files)} files to {TRAIN_FOLDER}")
print(f"✅ Moved {len(test_files)} files to {TEST_FOLDER}")
