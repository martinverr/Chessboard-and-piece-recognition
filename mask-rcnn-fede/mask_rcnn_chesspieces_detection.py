# scarica git clone https://github.com/z-mahmud22/Mask-RCNN_TF2.14.0.git
# segui l'installazione
# creazione di una cartella log dove vengono messi i dump del modello

# imports
from os import listdir
import json
import random
import skimage
from matplotlib.patches import Rectangle
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

from mrcnn.config import Config
from mrcnn.model import MaskRCNN

class ChessboardDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define classes
          # re -> k | regina -> q | alfiere -> b | torre -> t | cavallo -> n | pedone -> p
          # in questo caso se è di un colore ha le lettere piccole se è dell'altro ha le lettere grandi ma al momento non ci interessa
        self.add_class("dataset", 1, "k")
        self.add_class("dataset", 2, "q")
        self.add_class("dataset", 3, "b")
        self.add_class("dataset", 4, "r")
        self.add_class("dataset", 5, "n")
        self.add_class("dataset", 6, "p")

        # ho 4 cartelle: due per train e due per test e a seconda del valore is_train si segue una strada diversa
        if is_train:
          images_dir = dataset_dir + '/train_img/'
          annotations_dir = dataset_dir + '/train_annotation/'
        if not is_train:
          images_dir = dataset_dir + '/test_img/'
          annotations_dir = dataset_dir + '/test_annotation/'


		    # find all images
        for filename in listdir(images_dir):
            print(filename)
			  # extract image id
            image_id = filename[:-4]
			  #print('IMAGE ID: ',image_id)
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.json'
			  # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids = [0,1,2,3,4,5,6])

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
		# load and parse the file
        data = open(filename)
      # deserializing the data
        data = json.load(data)
        pieces = data['pieces']
		# extract each bounding box
        boxes = list()
        for box in pieces:
            name = box['piece'].lower()   #Add label name to the box list
            x,y,w,h = box['box']
            xmin = int(x)
            ymin = int(y)
            xmax = int(x+w)
            ymax = int(y+h)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
		# extract image dimensions
        width = int(1200)
        height = int(800)
        return boxes, width, height

    def load_mask(self, image_id):
		# get details of image
        info = self.image_info[image_id]
		# define box file location
        path = info['annotation']
        #return info, path


		# load json
        boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            # self.add_class("dataset", 1, "k")
            # self.add_class("dataset", 2, "q")
            # self.add_class("dataset", 3, "b")
            # self.add_class("dataset", 4, "r")
            # self.add_class("dataset", 5, "n")
            # self.add_class("dataset", 6, "p")
            # box[4] will have the name of the class
            if (box[4] == 'k'):
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(1)
            elif(box[4] == 'q'):
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(2)
            elif(box[4] == 'b'):
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(3)
            elif(box[4] == 'r'):
                masks[row_s:row_e, col_s:col_e, i] = 4
                class_ids.append(4)
            elif(box[4] == 'n'):
                masks[row_s:row_e, col_s:col_e, i] = 5
                class_ids.append(5)
            elif(box[4] == 'p'):
                masks[row_s:row_e, col_s:col_e, i] = 6
                class_ids.append(6)

        return masks, asarray(class_ids, dtype='int32')

        def image_reference(self, image_id):
          info = self.image_info[image_id]
          return info['path']

# define a configuration for the model
class ChessConfig(Config):
	# define the name of the configuration
	NAME = "chess_cfg"
	# number of classes (background + 6 pices)
	NUM_CLASSES = 1 + 6
	# number of training steps per epoch
	STEPS_PER_EPOCH = 75

	IMAGE_MIN_DIM  = 800 
	
	GPU_COUNT = 1


class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "chess_cfg"
	# number of classes (background + 6 fruits)
	NUM_CLASSES = 1 + 6
	# simplify GPU config
	GPU_COUNT = 1

	IMAGES_PER_GPU = 1

if __name__ == "__main__":

    # TRAIN set
    dataset_dir='dataset'

    train_set = ChessboardDataset()
    train_set.load_dataset(dataset_dir, is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    # TEST/VAL set
    test_set = ChessboardDataset()
    test_set.load_dataset(dataset_dir, is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))

    ###### prova per vedere se si vedono le box
    num=random.randint(0, len(train_set.image_ids))
    # define image id
    image_id = num
    # load the image
    image = train_set.load_image(image_id)
    # load the masks and the class ids
    mask, class_ids = train_set.load_mask(image_id)
    # extract bounding boxes from the masks
    bbox = extract_bboxes(mask)
    # display image with masks and bounding boxes
    display_instances(image, bbox, mask, class_ids, train_set.class_names)

    ##### config del modello ##########
    # prepare config
    config = ChessConfig()
    config.display()

    # define the model
    model = MaskRCNN(mode='training', model_dir="logs", config=config)
    # LOAD weights (mscoco) and exclude the output layers
    model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])


    # TRAIN weights (output layers or 'heads')
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')


    #### INFERENCE #####
    # create config
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)
    # LOAD model weights
    model.load_weights('mask_rcnn_chess_cfg_0003.h5', by_name=True)

    fruit_img = skimage.io.imread("dataset/test_img/0060.png")
    detected = model.detect([fruit_img])[0] 

    #### plot img ####
    pyplot.imshow(fruit_img)
    ax = pyplot.gca()
                # self.add_class("dataset", 1, "k")
                # self.add_class("dataset", 2, "q")
                # self.add_class("dataset", 3, "b")
                # self.add_class("dataset", 4, "r")
                # self.add_class("dataset", 5, "n")
                # self.add_class("dataset", 6, "p")
    class_names = ['k', 'q', 'b','r','n','p']
    class_id_counter=1
    for box in detected['rois']:
        #print(box)
    #get coordinates
        detected_class_id = detected['class_ids'][class_id_counter-1]
        #print(detected_class_id)
        #print("Detected class is :", class_names[detected_class_id-1])
        y1, x1, y2, x2 = box
        #calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        #create the shape
        ax.annotate(class_names[detected_class_id-1], (x1, y1), color='black', weight='bold', fontsize=10, ha='center', va='center')
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
    #draw the box
        ax.add_patch(rect)
        class_id_counter+=1
    #show the figure
    pyplot.show()


