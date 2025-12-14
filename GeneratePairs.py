# This program generates positive or negative pairs of images and labels them accordingly.
import os
import random
import csv
import glob

def generate_image_pairs(input_dir, output_csv="lfw_pairs.csv"):
    people_dict = {}

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npy'):
                person_name = os.path.basename(os.path.dirname(os.path.join(root, file)))
                if person_name not in people_dict:
                    people_dict[person_name] = []
                people_dict[person_name].append(os.path.join(root, file))
  
    multi_img_people = [name for name, files in people_dict.items() if len(files) > 1]
    num_pairs = len(multi_img_people) //2
    all_people_names = list(people_dict.keys())
    if not multi_img_people:
        print("Error: No people with >1 image found. Cannot create positive pairs.")
        return
    
    pairs_data = []

    target_positive_pairs = num_pairs // 2
    for _ in range(target_positive_pairs):  
        name = random.choice(multi_img_people)
        img1, img2 = random.sample(people_dict[name], 2)
        pairs_data.append([img1, img2, 1])

    for _ in range(num_pairs - target_positive_pairs):
        name1, name2 = random.sample(all_people_names, 2)
        img1 = random.choice(people_dict[name1])
        img2 = random.choice(people_dict[name2])
        pairs_data.append([img1, img2, 0])
    
    random.shuffle(pairs_data)
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["img1_path", "img2_path", "label"]) # Header
        writer.writerows(pairs_data)

if __name__ == "__main__":
    #lfw_dir = os.path.join('TrainingSet', 'lfw-deepfunneled', 'lfw-deepfunneled')
    generate_image_pairs(input_dir='TrainingSet', output_csv=os.path.join('TrainingSet', 'lfw_pairs.csv'))