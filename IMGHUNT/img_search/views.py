from django.shortcuts import render, redirect
from .forms import ImageForm, ContactForm
from .models import Image
from django.conf import settings
from django.http import HttpResponse
from django.core.mail import send_mail, BadHeaderError

import uuid
import glob
import os

import cv2
import numpy as np
import pickle

######################################################################
def main(request):
    cleanLastSearch() 
    # return render(request, 'img_search/main.html')
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            cleanLastSearch()  
            form.save()
            return render(request, 'img_search/searchImage.html', { 'currentimage' : getCurrentImageWithPath(), 'imagelist' : getResultImages()  })
            # return render(request, 'img_search/searchImage.html', { 'form': form  })
    else:
        form = ImageForm()
    
    return render(request, 'img_search/main.html', { 'form': form })


def about(request):
    return render(request, 'img_search/about.html')

def layout(request):
    return render(request, 'img_search/layout.html') 

def landing(request):
    return render(request, 'img_search/landing.html')  

def contactus(request):
    if request.method == 'GET':
        form = ContactForm()
    else:
        form = ContactForm(request.POST)
        if form.is_valid():
            subject = form.cleaned_data['subject']
            from_email = form.cleaned_data['from_email']
            message = form.cleaned_data['message']
            try:
                send_mail(subject, message, from_email, ['admin@example.com'])
            except BadHeaderError:
                return HttpResponse('Invalid header found!')
            return redirect('success')
    return render(request, "img_search/contactus.html", {'form': form})

def successView(request):
    return HttpResponse('Success! Thank you for your message.')

#def handler404(request,exception):
#    message = ["Oops!! You must have typed wrong URL, please use the navigation bar above.",
#                "404 Page not found error!!, Go back to HOME",
#                "What are you looking for? Go back to HOME."
#                ]
#    context = {
#        'message': message[randint(0,2)]
#        }
#    return render(request, "404.html", context, status=404)

#def error_500(request):
#        data = {}
#        return render(request,'img_search/500.html', data)
 

def searchImage(request):
    return render(request, 'img_search/searchImage.html', { 'currentimage' : getCurrentImageWithPath(), 'imagelist' : getResultImages()  })

def getResultImages():
    imageList = []

    ### Dummy data ##################
    # imageList.append('/media/images/dataset/100101.png')
    # imageList.append('/media/images/dataset/100800.png')
    #################################

    path = settings.MEDIA_ROOT + "\images"
    queryImg    = path + "\current\\" + getCurrentImage() 
    featureFile = path + "\index\index.pkl"
    imgDataset  = path + "\dataset"

    queryImg = queryImg.replace("\\","\\\\")
    imgDataset = imgDataset.replace("\\","\\\\")
    featureFile = featureFile.replace("\\","\\\\")

    print("LOG: Query Image:", queryImg)
    print("LOG: Feature file:", featureFile)
    #print("Dataset", imgDataset)

    # load the query image and show it
    # print("->>>>>>>>>Q", queryImg)
    queryImage = cv2.imread(queryImg)
    #queryImage = cv2.imread("D:/Projects/IMGHUNT/media/images/current/100502.png")
    # print("->>>>>>>>>QI", queryImage, queryImg)

    # Extracting features
    features = feature_extraction(imgDataset)
    # Saving extracted features
    save(features, featureFile)

    # finding 3D RGB histogram
    desc = RGBHistogram([8,8,8])
    queryFeatures = desc.describe(queryImage)

    #print("queryFeatures", queryFeatures )
    
    # loading off the index from the disk
    f = open(featureFile, 'rb')
    index = pickle.load(f)
    # f.close()

    #performing search
    searcher = Searcher(index)
    results = searcher.search(queryFeatures)

    # print("results", results)  ## All the results
    result_size = 18
    # print(results[0:result_size])         ## Select just the top 18 results

    for img_val in range(0, result_size):
        # print(results[img_val])
        imageList.append("/media/images/dataset/" + results[img_val][1])  ## Get the path formed out of the resulted images

    print("LOG: Resulted Image List", imageList)
    return imageList

### Method to retrive the current image searched for
def getCurrentImage():
    path = settings.MEDIA_ROOT + "\images\current"
    files = [f for f in glob.glob(path + "**\*.*")]
    for fl in files:
        currentFile = os.path.basename(fl)
        break  ## We know for sure there will be one and only one file here :-)
    return currentFile

### Method to retrive the current image searched for with relative path
def getCurrentImageWithPath():
    # path = settings.MEDIA_ROOT + "\images\current"
    # files = [f for f in glob.glob(path + "**\*.*")]
    # for fl in files:
    #     currentFile = os.path.basename(fl)
    #     break  ## We know for sure there will be one and only one file here :-)
    ###TODO: Write an exception handling here if no files etc.
    return '/media/images/current/' + getCurrentImage()

### Remove any file(s) from the current directory
def cleanLastSearch():
    print("LOG: Cleaning the search directory...")
    path = settings.MEDIA_ROOT + "\images\current"
    files = [fl for fl in glob.glob(path + "**\*.*")]
    for fl in files:
        print("LOG: Removing file ", fl)
        os.remove(fl)  ## WARNING: Enable this one 
        print("LOG: File removed.")
    print("LOG: Cleaning completed successfully!!")

################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 1: Image Descriptor
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################
class RGBHistogram:
    """
    Image descriptor using color histogram.
    :param bins: list
        Histogram size. 1-D list containing ideal values
        between 8 and 128; but you can go up till 0 - 256.
    Example:
        >>> histogram = RGBHistogram(bins=[32, 32, 32])
        >>> feature_vector = histogram.describe(image='folder/image.jpg')
        >>> print(feature_vector.shape)
    """

    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        """
        Color description of a given image
        compute a 3D histogram in the RGB color space,
        then normalize the histogram so that images
        with the same content, but either scaled larger
        or smaller will have (roughly) the same histogram
        :param image:
            Image to be described.
        :return: flattened 3-D histogram
            Flattened descriptor [feature vector].
        """
        # print(">>>>>>>", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=self.bins, ranges=[0, 256] * 3)
        hist = cv2.normalize(hist, dst=hist.shape)
        return hist.flatten()


################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 2: Indexing
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################
# key - image file name, value - computed feature vector/descriptor
def feature_extraction(dataset):
    features_extracted = {}
    descriptor = RGBHistogram(bins=[8, 8, 8])

    # print("MSG: I am at feature_extraction", dataset)
    # print(glob.glob(os.path.join(dataset, '*.jpg|png$')))
    # print(os.listdir(dataset))
    # for filename in glob.glob(os.path.join(dataset, '*.jpg|png$')):
    for filename in os.listdir(dataset):  
        # e.g. places/eiffel_tower.jpg => eiffel_tower
        # img_name = os.path.basename(filename).split('.')[0]
        flname = dataset + "\\" + filename
        # filename = dataset + filename
        # print("IMG: ", filename, flname)
        image = cv2.imread(flname)
        #print("******", image, filename)
        feature = descriptor.describe(image)
        # key - image name, value - feature vector
        features_extracted[filename] = feature
    
    #### DEBUG Reasons
    # filename='C:\\Users\\User\\Desktop\\ImageSearch\\avengers4.jpg'
    # image = cv2.imread(filename)
    # feature = descriptor.describe(image)
    # # key - image name, value - feature vector
    # img_name = os.path.basename(filename).split('.')[0]
    # # features_extracted[img_name] = feature
    # features_extracted['avengers4.jpg'] = feature

    # filename='C:\\Users\\User\\Desktop\\ImageSearch\\harry01.jpg'
    # image = cv2.imread(filename)
    # feature = descriptor.describe(image)
    # # key - image name, value - feature vector
    # img_name = os.path.basename(filename).split('.')[0]
    # features_extracted[img_name] = feature

    # filename='C:\\Users\\User\\Desktop\\ImageSearch\\Thanos01.jpg'
    # image = cv2.imread(filename)
    # feature = descriptor.describe(image)
    # # key - image name, value - feature vector
    # img_name = os.path.basename(filename).split('.')[0]
    # features_extracted[img_name] = feature
    ##################
    return features_extracted


# Writing the index to disk
def save(obj, path):
    # print(">>>>>>>>>", path)
    if not os.path.isfile(path):
        # print("#####################################")
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f,protocol = pickle.HIGHEST_PROTOCOL)
        #pickle.dump(obj, f)
        f.close()


################################################################################################
# +———————————————————————————————————————————————————————————————————————————————————————————+
# | Step 3: Searching
# +———————————————————————————————————————————————————————————————————————————————————————————+
################################################################################################
class Searcher:
    def __init__(self, features):
        self.features = features

    #@staticmethod
    def chi_squared(self, H_a, H_b,eps = 1e-10):
        # compute the chi-squared distance
        #eps = 1e-10
        dist = 0.5 * np.sum([pow(a - b, 2) / (a + b + eps)
                              for (a, b) in zip(H_a, H_b)])
        # return the chi-squared distance
        return dist

    def search(self, query):
        results = {}

        for name, feature in self.features.items():
            dist = self.chi_squared(query, feature)
            results[name] = dist

        results = sorted([(d, n) for n, d in results.items()])
        return results

    
##############END OF CODE#########################################
##################################################################