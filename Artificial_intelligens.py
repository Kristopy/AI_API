

#https://expressexpense.com/large-receipt-image-dataset-SRD.zip

import numpy as np
#Predicting data: 
label_legend_inverted = {'1':'spam', '0':'ham'}


def predict(label_legend_inverted):
    labeled_preds = [{f'{label_legend_inverted[str(i)]}': x} for i, x in enumerate(2)]
    return labeled_preds


print(predict(label_legend_inverted))
