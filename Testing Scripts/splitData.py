import os
import random
import shutil
from itertools import islice

outputFolderPath = 'Dataset/SplitData'
inputFolderPath = 'Dataset/All'
splitRatio = {"train":0.7, "val": 0.2, "test":0.1}
classes = ['fake','real']

try:
    shutil.rmtree(outputFolderPath)
except OSError as e:
    os.makedirs(outputFolderPath)

# Directories to create
os.makedirs(f'{outputFolderPath}/train/images',exist_ok=True)
os.makedirs(f'{outputFolderPath}/train/labels',exist_ok=True)
os.makedirs(f'{outputFolderPath}/val/images',exist_ok=True)
os.makedirs(f'{outputFolderPath}/val/labels',exist_ok=True)
os.makedirs(f'{outputFolderPath}/test/images',exist_ok=True)
os.makedirs(f'{outputFolderPath}/test/labels',exist_ok=True)

# Get the names
listNames = os.listdir(inputFolderPath)
uniqueNames = []

for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))

# Shuffle
random.shuffle(uniqueNames)

# Get the no. of images for each folder
lenData = len(uniqueNames)
lenTrain = int(lenData*splitRatio["train"])
lenVal = int(lenData*splitRatio["val"])
lenTest = int(lenData*splitRatio["test"])

# Putting remaining images in train
if lenData != (lenTrain+lenVal+lenTest):
    remaining = lenData - (lenTrain+lenVal+lenTest)
    lenTrain += remaining

# Split the list
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input,elem)) for elem in lengthToSplit]
print(f'Total images: {lenData}\n Split: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

# Copy the files
sequence = ["train", "val", "test"]
for i,out in enumerate(Output):
    for fileName in out:
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg',f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputFolderPath}/{fileName}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

print("Split Process completed...")

# Creating data.yaml files

dataYaml =f'path: /home/pi/Desktop/siddhant/research/spoofingDetection/Dataset/SplitData\n\
train: train/images\n\
val: val/images\n\
test: test/images\n\
\n\
nc : {len(classes)}\n\
names: {classes}'

f = open(f'{outputFolderPath}/data.yaml', 'a')
f.write(dataYaml)
f.close()
print("Data.yaml created successfully...")