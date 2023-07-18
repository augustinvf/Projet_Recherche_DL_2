import torch

def self_supervised_training(device, model, train_dataloader_self_supervised, criterion_ss, optimizer_ss, scheduler_ss):
    sum_loss_ss = 0
    for mini_batch, _ in train_dataloader_self_supervised :

        # reinitialization of the gradients
        optimizer_ss.zero_grad()

        # self-supervised phase

        # .to(device)
        augmented_image1 = mini_batch[0].to(device)
        augmented_image2 = mini_batch[1].to(device)

        # forward propagation for both images
        y_hat_1 = model(augmented_image1, "self-supervised")
        y_hat_2 = model(augmented_image2, "self-supervised")

        # loss calculation
        loss_ss = criterion_ss(y_hat_1, y_hat_2)
        sum_loss_ss += loss_ss.detach()

        # backward propagation
        loss_ss.backward()
        optimizer_ss.step()
    
    scheduler_ss.step()

    return sum_loss_ss

def supervised_training(device, model, train_dataloader_supervised, criterion_su, optimizer_su, scheduler_su):
    sum_loss_su = 0
    accuracy = 0
    for mini_batch, labels in train_dataloader_supervised :

        # reinitialization of the gradients
        optimizer_su.zero_grad()

        # supervised phase
        image_without_augmentation = mini_batch.to(device)
        labels = labels.to(device)

        y_hat = model(image_without_augmentation, "supervised")

        accuracy += torch.sum(torch.eq(torch.argmax(y_hat, axis = 1), labels))

        loss_su = criterion_su(y_hat, labels)
        sum_loss_su += loss_su.detach()

        # backward propagation
        loss_su.backward()
        optimizer_su.step()
    
    scheduler_su.step()

    return sum_loss_su, accuracy