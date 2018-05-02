import os

# IMPORTANT : this file needs to be run from : InfoGAN-Pytorch/data/
# This file also assumes that you have previously placed rendered_chairs.tar
#    in the InfoGAN-Pytorch/data/raw/ directory and that you have extracted there
root_dir = os.path.join("raw", "rendered_chairs")
new_dir = "3Dchairs"
all_chair_folders = os.listdir(root_dir)

for chair_folder in all_chair_folders:

	all_images = os.listdir(os.path.join(root_dir, chair_folder, "renders"))

	for img_file in all_images:

		old_name_and_location = os.path.join(root_dir, chair_folder, "renders", img_file)
		new_name_and_location = os.path.join(new_dir, chair_folder+"_"+img_file)
		os.rename(old_name_and_location, new_name_and_location)

