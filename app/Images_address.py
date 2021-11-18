
#Image converter from png to json file for reading in model. 
#Vectorizing the png file, reshaping to 28,28 and normalizing.

def Image_Convert(IMAGES_PATH):
    import cv2
    from fastapi import HTTPException
    try:
        image = cv2.imread(str(IMAGES_PATH), cv2.IMREAD_COLOR)
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_GRAY = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)

        dim = (28,28)
        resized = cv2.resize(image_GRAY, dim, interpolation=cv2.INTER_AREA)
        imagem = cv2.bitwise_not(resized)
        formatted_img = imagem/255
        return formatted_img.tolist()
    except:
        raise HTTPException(status_code=404, detail=f'Path not found or filename incorrect {str(IMAGES_PATH)}')

if __name__ == '__main__':
    Image_Convert()