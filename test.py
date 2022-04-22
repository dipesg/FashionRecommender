import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
from src.train import Training
import logger

class Testing:
    def __init__(self):
        # Initializing the logger object
        self.file_object = open("./Logs/test_logs.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.feature_list = np.array(pickle.load(open('./pickleFile/embeddings.pkl','rb')))
        self.filenames = pickle.load(open('pickleFile/filenames.pkl','rb'))
        
    
            
    def test_image(self, path):
        try:
            feature_list = self.feature_list
            model = Training().train_cnn()
            normalized_result = Training().normalize_image(path, model)
            neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
            neighbors.fit(feature_list)
            distances,indices = neighbors.kneighbors([normalized_result])
            return indices
            
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the test_image function!! Error:: %s' % ex)
            raise ex
        
        
    def show_result(self, indices):
        try:
            #distance, indices = self.test_image()
            for file in indices[0][1:6]:
                temp_img = cv2.imread(self.filenames[file])
                cv2.imshow('output',cv2.resize(temp_img,(512,512)))
                cv2.waitKey(0)
                
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the show_result function!! Error:: %s' % ex)
            raise ex
        
if __name__ == '__main__':
    test = Testing()
    indices = test.test_image('./test_images/shirt1.jpg')
    test.show_result(indices)
        
