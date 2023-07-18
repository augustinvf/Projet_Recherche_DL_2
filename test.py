import torch

def test_fct(device, model, test_dataloader):
    model.eval()

    total_tests = 0
    positive_tests = 0

    for mini_batch, labels in test_dataloader :

        image_without_augmentation = mini_batch.to(device)
        labels = labels.to(device)

        with torch.no_grad() :
            y_hat = model(image_without_augmentation, "supervised")

        total_tests += labels.shape[0]
        positive_tests += torch.sum(torch.eq(torch.argmax(y_hat, axis = 1), labels))

    final_accuracy = positive_tests / total_tests
    return final_accuracy
