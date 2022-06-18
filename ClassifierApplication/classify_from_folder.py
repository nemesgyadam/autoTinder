import sys
import tqdm
import os
from matplotlib import pyplot as plt
import argparse
from tqdm import tqdm
import readchar
import cv2
import shutil
import random
import string
import imutils


def parse_args(args):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--categories', help='The divided by a comma.', default='NOT_CAR,CAR')
    # parser.add_argument('--in_folder', help='The folder of the incoming images', default='/ml/res/IS_CAR_NOT_CAR/TO_PROCESS/PGergo/')
    # parser.add_argument('--out_folder', help='The folder of the outgoing images', default='/ml/res/IS_CAR_NOT_CAR/')
    
    parser.add_argument('--categories', help='The divided by a comma.', default='-,+')
    parser.add_argument('--in_folder', help='The folder of the incoming images', default='C:/Code/TINDER/resources/faces/unlabeled')
    parser.add_argument('--out_folder', help='The folder of the outgoing images', default='C:/Code/TINDER/resources/faces/nemes/')

    return parser.parse_args(args)

def correct_dir(d):
    return d if d[-1] == '/' else d+'/'

def print_information(categories):
    print('--------------------------------------------------------')
    print('INFO')
    print("    - Press left-key or 'a' or '1' to classify",categories[0])
    print("    - Press right-key or 'd' or '2' to classify",categories[1])
    if len(categories)>2:
        print("    - Press up-key or 'w' or '3' to classify",categories[2])
    if len(categories)>3:
        print("    - Press down-key or 's' or '4' to classify",categories[3])
    print('')
    print("    - Press space to move the image to bin (idle) folder.")
    print("    - Press backspace to redo classification.")
    print("    - Press q or ctrl+c to quit safely.")
    print('--------------------------------------------------------')

def mkdirs(root, dirs):
    root = correct_dir(root)
    for d in dirs:
        if not os.path.exists(root+d):
            os.makedirs(root+d)

def move_file(from_file, to_file):
    shutil.move(from_file, to_file)

def generate_filename(N=25):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))

def classify(args,key_codes):

	filenames = os.listdir(correct_dir(args.in_folder))

	assert len(filenames)>0, "Source folder is empty. Set up custom source folder or fill the default one!" 

	categories = args.categories.split(',')
	key_codes=key_codes[0:len(categories)]
	mkdirs(args.out_folder, categories)
	mkdirs(args.out_folder, ['BIN'])

	print_information(categories)

	cv2.namedWindow("classifier")

	history = []
	i = 0

	with tqdm(total=len(filenames)) as pbar:
		while i < len(filenames):
			try:
				img = cv2.imread(correct_dir(args.in_folder) + filenames[i])
				img = imutils.resize(img, width=500, height=500)
				cv2.imshow("classifier", img)
			except:
				try:
					cv2.destroyAllWindows()
				except:
					pass
				new_filename = generate_filename() + '.' + filenames[i].split('.')[-1]
				departure = correct_dir(args.in_folder) + filenames[i]
				destination = correct_dir('bin') + new_filename
				move_file(departure, destination)
				history.append([departure, destination])
				continue

			# Read char
			k = None
			while (1):
				k = cv2.waitKeyEx(0)
				if k != -1:
					break

			change=0

			if k == 32:

				# space --> delete
				new_filename = generate_filename() + '.' + filenames[i].split('.')[-1]
				departure = correct_dir(args.in_folder) + filenames[i]
				destination = correct_dir(args.out_folder) + correct_dir('BIN') + new_filename
				move_file(departure, destination)
				history.append([departure, destination])
				change=1

			elif k == 113 or k == 3:
				# q or ctrl+c --> Quit
				print(' EXIT')
				cv2.destroyAllWindows()
				pbar.close()
				break

			elif k == 8:
				#UNDO
				try:
					last_pic_info = history.pop()
					move_file(last_pic_info[1], last_pic_info[0])
					change=-1
				except:
					print("No previous image(s) available")
			else:
				category_selected = False
				# Act
				for j in range(len(categories)):
					if k in key_codes[j]:
						cat_id = j
						category_selected = True

				if category_selected:
					new_filename = generate_filename() + '.' + filenames[i].split('.')[-1]
					departure = correct_dir(args.in_folder) + filenames[i]
					destination = correct_dir(args.out_folder) + correct_dir(categories[cat_id]) + new_filename
					move_file(departure, destination)
					history.append([departure, destination])
					change=1
				
			i += change
			pbar.n += change
			pbar.refresh()

	cv2.destroyAllWindows()


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    key_codes = [[ord('a'),2424832,49],[ord('d'),2555904,50],[ord('w'),2490368,51],[ord('s'),2621440,52]]
    classify(args,key_codes)


if __name__ == "__main__":
    main()