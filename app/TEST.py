

if __name__ == '__main__':
    import pathlib
    from AI_models import AIModel
    import random
    import numpy as np
    from Images_address import Image_Convert

    BASE_DIR = pathlib.Path().resolve(__file__)
    DATASETS_DIR = BASE_DIR / 'Datasets'
    DATASETS_NUM_REC_DIR = DATASETS_DIR / 'Datasets_NUM_REC'
    EXPORT_NUM_REC_DIR = DATASETS_NUM_REC_DIR / 'Exports'
    MODEL_NUM_REC_EXPORT_PATH = EXPORT_NUM_REC_DIR / 'Num_Rec_Model.h5'
    METADATA_NUM_REC_EXPORT_PATH = EXPORT_NUM_REC_DIR / 'Num_Rec_Metadata.json'
    IMAGE_DIR = DATASETS_NUM_REC_DIR / 'Images_convert'
    IMAGE_PATH = IMAGE_DIR / 'Number_2.png'

    AI_MODEL = AIModel(
        model_path=MODEL_NUM_REC_EXPORT_PATH,
        metadata_path=METADATA_NUM_REC_EXPORT_PATH
    )

    metadata = AI_MODEL.get_metadata()
    model = AI_MODEL.get_model()


   
    X_test = metadata['X_test']
    y_test = metadata['y_test']
    Imp_img = Image_Convert()

    #random_idx = random.randint(0, len(X_test))
    predictions = model.predict([Imp_img])

    print('Predicted results: ', np.argmax(predictions), '\n' )


    # metadata = pd.DataFrame(metadata['X_test'][0])
    # new = metadata.round(1)
    # max = metadata.max(axis=0)
    # print(max)
    # new.to_csv("Test.csv", index=False)

