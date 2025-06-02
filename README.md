# NEURAL NETWORK FOR OBJECT RECOGNITION IN SATELLITE IMAGES V 0.3

The neural network model includes R-CNN, Siamese, and U-net architectures. It is designed for similar image retrieval, object recognition, and semantic segmentation.
Detailed documentation, sample training datasets, and step-by-step guides on creating custom datasets and using the model will be available soon. Currently, all interface elements and comments are in Russian, with English translations planned for future updates.
  
## 🗺️ API Google Maps

Allows downloading satellite map tiles to create datasets for training from Google Maps

- Enter coordinates (latitude & longitude) to download map tiles
- Auto-converts to tile x-y format
- Adjust zoom level for different resolutions

## ♊ Siam

Finds matching satellite images

- Provide your dataset (with grouped similar images)
- Upload an image to compare
- Get a collage of image pairs with the model's prediction

## 🚀 R-CNN

Detects objects in satellite images

- Create labeled images with bounding boxes (use LabelImg). Note: Only building detection is currently supported
- Upload a satellite image to analyze
- Get results: Marked-up image with detected objects, object types and confidence levels

  ## 🎨 U-NET

Performs semantic segmentation

- Create image masks using CVAT or similar tools and provide matching image-mask pairs
- Upload an image to analyze
- Receive results: side-by-side display of input image and segmentation result

- ## ♻️ Mask XML to PNG helper

Converts XML annotation masks to PNG format

- Upload an XML annotation file (must be named 'annotations.xml')
- Receive output mask in PNG format
