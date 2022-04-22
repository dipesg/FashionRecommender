import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import logger

class Training:
    def __init__(self):
        # Initializing the logger object
        self.file_object = open("./Logs/train_logs.txt", 'a+')
        self.log_writer = logger.App_Logger()
     
        
    def train_cnn(self):
        try:
            self.model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
            self.model.trainable = False
            self.model = tensorflow.keras.Sequential([
                self.model,
                GlobalMaxPooling2D()
            ])
            return self.model
        
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the model function!! Error:: %s' % ex)
            raise ex


    def normalize_image(self, img_path, model):
        try:
            img = image.load_img(img_path,target_size=(224,224))
            img_array = image.img_to_array(img)
            expanded_img_array = np.expand_dims(img_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img_array)
            result = model.predict(preprocessed_img).flatten()
            normalized_result = result / norm(result)
            return normalized_result
        
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the normalize_image function!! Error:: %s' % ex)
            raise ex
    
    
    def extract_features(self):
        try:
            model = self.train_cnn()
            filenames = []
            feature_list = []
            for file in os.listdir('./images'):
                filenames.append(os.path.join('./images',file))

            for file in tqdm(filenames):
                feature_list.append(self.normalize_image(file,model))  
            return filenames, feature_list
        
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the extract_features function!! Error:: %s' % ex)
            raise ex
        
    
    def save_pickle(self):
        try:
            filenames, feature_list = self.extract_features()
            pickle.dump(feature_list,open('pickleFile/embeddings.pkl','wb'))
            pickle.dump(filenames,open('pickleFile/filenames.pkl','wb'))
            
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the save_pickle function!! Error:: %s' % ex)
            raise ex
        
if __name__ == '__main__':
    train = Training()
    train.save_pickle()

