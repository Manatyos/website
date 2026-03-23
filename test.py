from script import predictrf
from script import predictlr
from script import predictsvr

imagepath = r"imagestorage/IMG_2754.jpg"
xrf = predictrf(imagepath)
print("RandomForest:", xrf)

imagepath = r"imagestorage/IMG_2754.jpg"
xlr = predictlr(imagepath)
print("LinearRegression:", xlr)

imagepath = r"imagestorage/IMG_2754.jpg"
xsvr = predictsvr(imagepath)
print("SVR:", xsvr)