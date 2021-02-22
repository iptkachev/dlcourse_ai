def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    count_objects = ground_truth.shape[0]
    
    tp = 0
    for i in range(prediction.shape[0]):
        if prediction[i]:
            tp += (prediction[i] == ground_truth[i])
            
    recall = tp / ground_truth.sum()
    precision = tp / prediction.sum()
    f1 = 2 * precision * recall / (precision + recall)
    
    accuracy = (prediction == ground_truth).sum() / count_objects
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    count_objects = ground_truth.shape[0]
    return (prediction == ground_truth).sum() / count_objects
