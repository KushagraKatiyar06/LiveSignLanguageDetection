from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os


#model_path = "asl_model.h5"  
#img_path = "asl_dataset/4/hand1_4_bot_seg_1_cropped.jpeg"


if not os.path.exists(img_path):
    raise FileNotFoundError(f"The file {img_path} does not exist.")


model = load_model(model_path)


img = image.load_img(img_path, target_size=(400, 400))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0 


prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]


plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis("off")
plt.show()