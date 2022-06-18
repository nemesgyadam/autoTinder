import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
import uuid
import shutil

def showImg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.rcParams["figure.figsize"] = (10,10)
    plt.imshow(img)
    plt.show()
    
def showMe(history):
    for key in history.history.keys():

        if 'val_' not in key and 'lr' != key:
            try:
                plt.clf()
                plt.plot(history.history[key])
                plt.plot(history.history['val_'+key])
                #plt.title(key)
                plt.ylabel(key)
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper left')
                plt.show()
            except:
                ...


def MyEval(model, generator, output = None, threshold = None, draw = True, save_images = False, version = 'dumb'):
    test_result_dir = '/ml/test_results/M0_isCar/'
    target_dir = os.path.join(test_result_dir,version)
    FN_dir = os.path.join(target_dir,'FN')
    FP_dir = os.path.join(target_dir,'FP')
    
    # try:
    #     print("Threshold: "+str(threshold))
    # except:
    #     ...
    
    if save_images:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if not os.path.exists(FN_dir):
            os.makedirs(FN_dir)
        if not os.path.exists(FP_dir):
            os.makedirs(FP_dir)


    correct = 0
    incorrect = 0
    
    if output == 1:
        classes = generator.vehicle_classes
    else:
        classes = generator.classes
        
    binary = len(classes) == 2    
    print("Binary: "+str(binary))
    if binary and threshold is None:
        print("Please define threshold! (0.5 setted)")
        threshold = 0.5
    matrix = np.zeros((len(classes),len(classes)))
    for i in range(len(generator)):
        if output is None:
            gts = generator[i][1]
            results = model(generator[i][0]).numpy()
        else:
            gts = generator[i][1][output]
            results = model(generator[i][0])[output].numpy()

        
        for j in range(len(gts)):
            if not binary:
                gt = np.argmax(gts[j])
            else:
                gt = int(gts[j])

            if not binary:
                result = np.argmax(results[j])
            else:
                result = 1 if results[j]>threshold else 0
            
            matrix[gt,result]+=1

            if gt==result:
                correct += 1
            else:
                incorrect+=1

                if save_images:
                    if gt == 1:
                        cv2.imwrite(os.path.join(FN_dir, str(uuid.uuid4())+'.jpg'), generator[i][0][j])
                    else:
                        cv2.imwrite( os.path.join(FP_dir, str(uuid.uuid4())+'.jpg'), generator[i][0][j])
    accuracy = round(correct/(correct+incorrect)*100)
    
    if output == 1:
        print("Evaluting Vehicle Category: {}%".format(accuracy))
    else:
        print("Evaluting Axle Category: {}%".format(accuracy))

    if draw:
        DrawConfusionMatrix(matrix,classes)
        
    return accuracy

def DrawConfusionMatrix(matrix, class_names = ["2","3","4","5"], draw = True):
    class_names = [element.split('_')[0] for element in class_names]
    plt.clf()
    cmap = sns.light_palette("#2ecc71", as_cmap=True)
    #ax = sns.heatmap(np.array(matrix),annot=matrix, xticklabels=class_names, yticklabels=class_names, cmap=cmap,  fmt='g')
    ax =sns.heatmap(np.array(matrix),annot=matrix, xticklabels=class_names, yticklabels=class_names, cmap=cmap,  fmt='g')

    
    plt.xlabel('Predictions', fontsize = 15)
    plt.ylabel('Ground Truths', fontsize = 15)
    plt.subplots_adjust(left=0.25)
    plt.subplots_adjust(bottom=0.25)
    if draw:
        plt.show()
    else:   
        return plt


def FolderTest(model, 
               path, 
               input_shape, 
               classes = ['2','3','4','5'],
               threshold = None,
               save_incorrect = False,
               version = 'dumb',
               TEST_RESULT_DIR ='/ml/test_results/axle_classifier/' 
               ):
    if save_incorrect:
        # CREATE IMAGE STURCTURE
        try:
            os.mkdir(os.path.join(TEST_RESULT_DIR, version))
        except:
            ...
        try:
            target_dir = os.path.join(TEST_RESULT_DIR, version, path.split('/')[-1])
            os.mkdir(target_dir)
            for f in range(2,6):
                    os.mkdir(os.path.join(target_dir,str(f)))
        except Exception as e:
            print("Folders already exits, overriding...")
            
    binary = len(classes) == 2    
    if binary and threshold is None:
        print("Please define threshold! (0.5 setted)")
        threshold = 0.5

    classes = os.listdir(path)
    files = []
    for c in classes:
        for f in os.listdir(os.path.join(path,c)):
            files.append(os.path.join(path,c,f))

    correct = 0
    incorrect = 0
    matrix = np.zeros((len(classes),len(classes)))
    for i in range(len(files)):
        gt = files[i].split('/')[-2]
        
        img_path = files[i]
        try:
            img =  cv2.imread(img_path)
            img =  cv2.resize(img, input_shape)
            img = np.array(img, dtype=np.float32)
            img /= 255
        except:
            print("Error reading image:", img_path)
            break
        #img /= 255
        result = model(np.expand_dims(img, axis=0))[0].numpy()[0]
        #print("Processing {} --> {}".format(files[i], result))

        try:
            gt = classes.index(gt)
        except:
            # SET CAR IF FILENAME NOT START WITH CLASSNAME
            gt = 1
        if not binary:
            result = np.argmax(result)
        else:
            result = 1 if result>threshold else 0
        matrix[int(gt),int(result)]+=1

        if gt==result:
            correct += 1
        else:
            incorrect+=1
            if save_incorrect:
                shutil.copy(files[i], os.path.join(target_dir,classes[result], files[i].split('/')[-1]))
    print("Accuracy {}%".format(round(correct/(correct+incorrect)*100)))
    DrawConfusionMatrix(matrix,classes).savefig(os.path.join(target_dir, 'confusion_matrix.png'),facecolor='white')
    

def FolderTestOld(model, path, input_shape, classes = ['2','3','4','5'], threshold = None):

    binary = len(classes) == 2    
    if binary and threshold is None:
        print("Please define threshold! (0.5 setted)")
        threshold = 0.5


    files = os.listdir(path)
    correct = 0
    incorrect = 0
    matrix = np.zeros((len(classes),len(classes)))
    for i in range(len(files)):
        gt = files[i].split('_')[0]
        
        img_path = os.path.join(path,files[i])
        img =  cv2.imread(img_path)
        img =  cv2.resize(img, input_shape)
        img = np.array(img, dtype=np.float32)
        #img /= 255
        result = model(np.expand_dims(img, axis=0))[0].numpy()[0]
        print("Processing {} --> {}".format(files[i], result))

        try:
            gt = classes.index(gt)
        except:
            # SET CAR IF FILENAME NOT START WITH CLASSNAME
            gt = 1
        if not binary:
            result = np.argmax(result)
        else:
            result = 1 if result>threshold else 0
        matrix[int(gt),int(result)]+=1

        if gt==result:
            correct += 1
        else:
            incorrect+=1
    print("{}%".format(round(correct/(correct+incorrect)*100)))
    DrawConfusionMatrix(matrix, classes)
    return matrix
