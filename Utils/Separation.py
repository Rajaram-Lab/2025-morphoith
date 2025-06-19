# %%

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def GetAccuracies(vectors, labels):
    """Given reduced feature vectors and their labels, run support-vector classification.

    Args:
        vectors (array): feature vectors reduced to 10 dimensions.
        labels (array): labels of the feature vectors (nuclear grade, vascular architectures).

    Returns:
        linear_accuracy: separation measure (SVC classification accuracy).
    """
    
    x_train, x_test, y_train, y_test = train_test_split(vectors, labels, 
                                                        test_size=0.5, random_state=111, 
                                                        stratify = labels, shuffle=True)
    linear = svm.SVC(kernel='linear', class_weight='balanced').fit(x_train, y_train)
    linear_pred = linear.predict(x_test)
    linear_accuracy = accuracy_score(y_test, linear_pred)

    return linear_accuracy
