import glob

readpath = 'E:/IP/dataset 256/HTC-1-M7/*.jpg'

labels = 'E:/IP/Labels/labels.csv'
objectClass = '1'

images = glob.glob(readpath)
labelfile = open(labels,'w')

for image in images:
    labelfile.write(image+','+objectClass+'\n')

labelfile.close()


readpath = 'E:/IP/dataset 256/iPhone-4s/*.jpg'
labels = 'E:/IP/Labels/labels.csv'
objectClass = '2'

images = glob.glob(readpath)
labelfile = open(labels,'a')  

for image in images:
    labelfile.write(image+','+objectClass+'\n')

labelfile.close()



readpath = 'E:/IP/dataset 256/iPhone-6/*.jpg'
labels = 'E:/IP/Labels/labels.csv'
objectClass = '3'

images = glob.glob(readpath)
labelfile = open(labels,'a')

for image in images:
    labelfile.write(image+','+objectClass+'\n')

labelfile.close()




readpath = 'E:/IP/dataset 256/LG Nexus 5x/*.jpg'
labels = 'E:/IP/Labels/labels.csv'
objectClass = '4'

images = glob.glob(readpath)
labelfile = open(labels,'a')  

for image in images:
    labelfile.write(image+','+objectClass+'\n')

labelfile.close()



readpath = 'E:/IP/dataset 256/Motorola Droid Maxx/*.jpg'
labels = 'E:/IP/Labels/labels.csv'
objectClass = '5'

images = glob.glob(readpath)
labelfile = open(labels,'a')  

for image in images:
    labelfile.write(image+','+objectClass+'\n')

labelfile.close()

readpath = 'E:/IP/dataset 256/Motorola Nexus 6/*.jpg'
labels = 'E:/IP/Labels/labels.csv'
objectClass = '6'

images = glob.glob(readpath)
labelfile = open(labels,'a')  

for image in images:
    labelfile.write(image+','+objectClass+'\n')

labelfile.close()


readpath = 'E:/IP/dataset 256/Motorola X/*.jpg'
labels = 'E:/IP/Labels/labels.csv'
objectClass = '7'

images = glob.glob(readpath)
labelfile = open(labels,'a')  

for image in images:
    labelfile.write(image+','+objectClass+'\n')

labelfile.close()



readpath = 'E:/IP/dataset 256/Samsung Galaxy Note 4/*.jpg'
labels = 'E:/IP/Labels/labels.csv'
objectClass = '8'

images = glob.glob(readpath)
labelfile = open(labels,'a')  

for image in images:
    labelfile.write(image+','+objectClass+'\n')

labelfile.close()


readpath = 'E:/IP/dataset 256/Samsung Galaxy S4/*.jpg'
labels = 'E:/IP/Labels/labels.csv'
objectClass = '9'

images = glob.glob(readpath)
labelfile = open(labels,'a')  

for image in images:
    labelfile.write(image+','+objectClass+'\n')

labelfile.close()


readpath = 'E:/IP/dataset 256/Sony Nex 7/*.jpg'
labels = 'E:/IP/Labels/labels.csv'
objectClass = '10'

images = glob.glob(readpath)
labelfile = open(labels,'a')  

for image in images:
    labelfile.write(image+','+objectClass+'\n')

labelfile.close()
