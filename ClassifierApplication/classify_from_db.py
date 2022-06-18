import sys
import os
import argparse
import readchar
import cv2
import shutil
import random
import pandas as pd
import mysql.connector



SQL_USER_NAME = 'user'
SQL_PASSWORD = 'jelszo99'
SQL_HOST = '192.168.11.200'
SQL_DB =  'training_data'
TABLE= 'car_training'
TAG_TABLE = 'car_training_car_tags_map'

IMAGE_FOLDER = '/Training/images/VERIFIED/'

highway_tag = '3e2e31bc-bc01-4d95-8ac9-f0be0dd31f0f'
parking_tag = 'e5f3bc10-4722-4269-bffb-187cf00c95bd'
mcu_tag     = '09f57be5-b1b4-448a-b003-529bf7755e38'
no_rider_tag= 'f9433ebd-4b47-4d5f-b690-b21b671c37e7'
tbd_tag     = 'c7590d0f-feec-4bb6-83da-5bcca5904c88'

tags = {'HIGHWAY':highway_tag, 'PARKING': parking_tag, 'MCU' : mcu_tag, 'NO_RIDER' : no_rider_tag, 'TBD' : tbd_tag}

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', help='The divided by a comma.', default='HIGHWAY,PARKING,MCU,NO_RIDER')
    parser.add_argument('--NAS', help='PATH TO NAS(MOUNTED NAME)', default='N:')
    parser.add_argument('--random', help='GET IMAGES IN RANDOM ORDER', action='store_true')

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
    print("    - Press 'p' to remove image.")
    print("    - Press space to skip image.")
    print("    - Press backspace to redo.")
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

def classify(args, key_codes, db, IMAGE_FOLDER):
    categories = args.categories.split(',')
    key_codes=key_codes[0:len(categories)]

    print_information(categories)

    cv2.namedWindow("classifier")

    history = []
    i = 0
    to_fix = None
    image_available = True
    while image_available:
        if to_fix is not None:
            filename = to_fix
            to_fix = None
        else:
            try:
                filename = get_image(db, args.random)
            except Exception as e:
                print(e)
                print("No more images available! GJ")
                image_available = False
        if filename in history:
            continue
        try:
            #print("Loading image: "+filename)
            img = cv2.imread(IMAGE_FOLDER + filename)
            img = cv2.resize(img, (500, 500))
            cv2.imshow("classifier", img)
        except:
            try:
                cv2.destroyAllWindows()
            except:
                pass
            
            history.append(filename)
            continue

        # Read char
        k = None
        while (1):
            k = cv2.waitKeyEx(0)
            if k != -1:
                break

        change=0

        if k == 32:
            history.append(filename)
            
            change=1

        elif k == 113 or k == 3:
            # q or ctrl+c --> Quit
            print(' EXIT')
            cv2.destroyAllWindows()

            break

        elif k == 8:
            #UNDO
            try:
                last_pic_info = history.pop()
                remove_tag(db, last_pic_info)
                to_fix = last_pic_info
                change=-1
            except:
                print("No previous image(s) available")
        elif k == ord('p'):
            tagId = tags[list(tags)[-1]]
            ################## MAGIC ##############################
            set_tag(db, filename, tagId)
            history.append(filename)
            change=1
        else:
            category_selected = False
            # Act
            for j in range(len(categories)):
                if k in key_codes[j]:
                    cat_id = j
                    category_selected = True

            if category_selected:
                tagId = tags[list(tags)[cat_id]]
                ################## MAGIC ##############################
                set_tag(db, filename, tagId)
                history.append(filename)
                change=1
        
        i += change


    cv2.destroyAllWindows()

def connect_to_sql():
    try:
        db = mysql.connector.connect(user=SQL_USER_NAME, password=SQL_PASSWORD,
        host=SQL_HOST, database=SQL_DB,
        auth_plugin='mysql_native_password')
        print("Connected to DB")
        return db
    except Exception as e:
        print(e)
        print("DB not available")
        quit()

def get_images(db, shuffle):
    cursor = db.cursor()

    #TAG REQUIRED FOR IMAGE
    query = '''select imageName from '''+SQL_DB+'.'+TABLE+''' ct3 where isDoubleChecked = '1' and imageName not in (
            select imageName from training_data.car_training ct left join training_data.car_training_car_tags_map  ct2 on ct.boxId = ct2.boxId where isDoubleChecked = '1' and (tagId = 'e5f3bc10-4722-4269-bffb-187cf00c95bd' or tagId = '3e2e31bc-bc01-4d95-8ac9-f0be0dd31f0f' or tagId = '09f57be5-b1b4-448a-b003-529bf7755e38' or tagId = 'f9433ebd-4b47-4d5f-b690-b21b671c37e7' or tagId = 'c7590d0f-feec-4bb6-83da-5bcca5904c88')
            )group by imageName order by verifiedAt DESC'''
    
    imagelist = pd.read_sql(query, db)['imageName'].tolist()
    if shuffle:
        random.shuffle(imagelist)
    return imagelist

def get_image(db, shuffle):
    return get_images(db, shuffle)[0]

def set_tag(db, imageName, tagId):
    try:
        cursor = db.cursor()
        #COLLECT BOXES OF IMAGE
        query = "select boxId from "+SQL_DB+'.'+TABLE+" ct where imageName='"+imageName+"'"
        boxIds = pd.read_sql(query, db)['boxId'].tolist()

        #ADD TAG TO IMAGE
        query = 'INSERT INTO '+SQL_DB+'.'+TAG_TABLE+' (boxId, tagId) VALUES '
        for boxId in boxIds:
            query+=('("'+boxId+'", "'+tagId+'"),')
        query = query[:-1]
        #print(query)
        cursor.execute(query)
        db.commit()
    except Exception as e:
        print(e)

def remove_tag(db, imageName):
    cursor = db.cursor()
    #COLLECT BOXES OF IMAGE
    query = "select boxId from "+SQL_DB+'.'+TABLE+" ct where imageName='"+imageName+"'"
    boxIds = pd.read_sql(query, db)['boxId'].tolist()

    #REMOVE TAG TO IMAGE
    query = 'DELETE FROM '+SQL_DB+'.'+TAG_TABLE+' WHERE boxId IN ('
    for boxId in boxIds:
        query+=('"'+boxId+'",')
    query = query[:-1]
    query+=')'
    #print(query)
    cursor.execute(query)
    db.commit()

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    key_codes = [[ord('a'),2424832,49],[ord('d'),2555904,50],[ord('w'),2490368,51],[ord('s'),2621440,52]]
    db = connect_to_sql()
    #image_list = get_images(db, db,args.random)

    classify(args,key_codes, db, args.NAS+IMAGE_FOLDER)


if __name__ == "__main__":
    main()