"""### Training tools"""

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import classification_report
import time
import tqdm


def logit_accuracy(logits, y_true):
    """This function computes the accuracy based on the logit outputs
    of the neural network (tensor) and the true y labels in integer form"""
    max_vals, y_pred = torch.max(logits, 1)
    acc = (y_pred == y_true).sum().item()/y_pred.size()[0]
    return acc, y_pred


def trainNN(model, model_folder_name, max_epochs, trainloader, valloader,
            testloader, lr, optimizer, criterion, patch_size, transform=False,
            patience=30, device="cpu", y_limit=0, save_to=None, regularizer=0,
            dropout=0, classification=True, supervision=True, pre_model=False):
    """This function trains a neural network (model) and then plot the loss
    diagrams on the training and validation sets through the epochs """
    
    # File and folder settings
    if save_to:
      save_to = str(os.path.join(save_to, model_folder_name))
      now = datetime.datetime.now() + datetime.timedelta(hours=3)
      fname = str(patch_size) + "_" + str(now.strftime("%Y%m%d-%H%M%S"))
      os.makedirs(os.path.join(save_to, fname))
      if not classification and not supervision:
          os.makedirs(os.path.join(save_to, fname, "figures"))

    # Initialisations
    epoch = 0
    epoch_loss_val = []
    epoch_acc_val = []
    epoch_loss_test = []
    epoch_acc_test = []
    epoch_loss_train = []
    epoch_acc_train = []
    countdown = patience

    # Training
    while epoch < max_epochs and countdown > 0:
        # Training
        epoch +=1
        batch_loss = []
        y_pred = []
        batch_acc = []
        model.train()
        # enumerate fetches a batch of the data for training!
        start = time.time()
        with tqdm.tqdm(total=len(trainloader)) as pbar:
            for i, data in enumerate(trainloader):
                # assign batched X,y to variables
                if pre_model:
                    inputs = pre_model.encode(data['tensors'].float().to(device))
                else:
                    inputs = data['tensors'].float().to(device)
                labels = data['labels'].long().to(device) if classification else \
                    data['labels'].float().to(device)
    
                # reset gradients for each batch
                # if we have different learning rates for the different parts of the
                # NN then a list of optimizers is fed to the function
                if type(optimizer) is list:
                    [optimizer[i].zero_grad() for i in range(len(optimizer))]
                else:
                    optimizer.zero_grad()
                # forward step
                logits = model(inputs)
                # compute loss and save it to list
                loss = criterion(logits, labels) + regularizer * model.l2_loss() if \
                       regularizer > 0 else criterion(logits, labels)
                batch_loss.append(loss.item())
                # backpropagate the loss
                loss.backward()
                ## update weights according to the selected optimizer
                # if we have different learning rates for the different parts of the
                # NN then a list of optimizers is fed to the function
                if type(optimizer) is list:
                    [optimizer[i].step() for i in range(len(optimizer))]
                else:
                    optimizer.step()
                if classification:
                    # calculate batch accuracy for this epoch
                    batch_acc.append(logit_accuracy(logits, labels)[0])
                pbar.update()
        print("\n\n")      
        if not classification and not supervision:
            try:
                dioni_composite = labels[0].detach().cpu().numpy()[[23, 11, 7], :, :]
                plt.title("True")
                plt.imsave(os.path.join(save_to, fname, "figures", str(epoch)+ "_True.png"),
                           dioni_composite.transpose(1,2,0)) 
            except ValueError:
                pass
            try:
                dioni_composite = logits[0].detach().cpu().numpy()[[23, 11, 7], :, :]
                plt.title("Rec")
                plt.imsave(os.path.join(save_to, fname, "figures", str(epoch)+ "_Rec.png"),
                           dioni_composite.transpose(1,2,0)) 
            except ValueError:
                pass
        end = time.time()
        # store each epochs performances for latter processing and plots
        epoch_loss_train.append(np.mean(batch_loss))
        if classification:
            epoch_acc_train.append(np.mean(batch_acc))
            print("Epoch {}\n===============\nTraining loss  : {:.4f}, Training accuracy  : {:.4f}".
                  format(epoch, np.mean(batch_loss), np.mean(batch_acc)))
        else:
            print("Epoch {}\n===============\nTraining loss  : {:.4f}".
                  format(epoch, np.mean(batch_loss)))
        print("Training epoch duration: {} sec ".format(str(
            np.round(end-start, 2))))

        if supervision:
            # Validation
            batch_loss_val = []
            batch_acc_val = []
            y_pred_val = []
            y_val=[]
    
            # Validation / Test
            model.eval()
            # no need to store gradients here (Validation purposes only)
            with torch.no_grad():
                for i, data in enumerate(valloader):
                    if pre_model:
                        inputs = pre_model.encode(data['tensors'].float().to(device))
                    else:
                        inputs = data['tensors'].float().to(device)
                    #inputs = data['tensors'].float().to(device)
                    labels = data['labels'].long().to(device)
                    logits = model(inputs.float())
                    loss_val = criterion(logits, labels)
                    batch_loss_val.append(loss_val.item())
                    batch_acc, batch_pred = logit_accuracy(logits, labels)
                    batch_acc_val.append(batch_acc)
                    y_pred_val.append(batch_pred)
                    y_val.append(data['labels'])
                epoch_loss_val.append(np.mean(batch_loss_val))
                if classification:
                    epoch_acc_val.append(np.mean(batch_acc_val))
                    print("Validation loss: {:1.4f}, Validation accuracy: {:1.4f}".
                            format(epoch_loss_val[-1], epoch_acc_val[-1]))
                else:
                    print("Validation loss: {:1.4f}".format(epoch_loss_val[-1]))
                          
                y_pred_val = np.concatenate([y_pred_val[i].to("cpu").numpy()
                                for i in range(len(y_pred_val))]).reshape(-1)
                y_val = np.concatenate([y_val[i].to("cpu").numpy()
                                for i in range(len(y_val))]).reshape(-1)
                labels = list(y_val)
                if classification:
                    clval = classification_report(y_val, y_pred_val, digits=3)
    
                #Test
                batch_loss_test = []
                batch_acc_test = []
                y_pred_test = []
                y_test = []
                for i, data in enumerate(testloader):
                    if pre_model:
                        inputs = pre_model.encode(data['tensors'].float().to(device))
                    else:
                        inputs = data['tensors'].float().to(device)
                    #inputs = data['tensors'].float().to(device)
                    labels = data['labels'].long().to(device)
                    logits = model(inputs.float())
                    loss_test = criterion(logits, labels)
                    batch_loss_test.append(loss_test.item())   
                    batch_acc, batch_pred = logit_accuracy(logits, labels)
                    batch_acc_test.append(batch_acc)
                    y_pred_test.append(batch_pred)
                    y_test.append(data['labels'])
                epoch_loss_test.append(np.mean(batch_loss_test))
                if classification:
                    epoch_acc_test.append(np.mean(batch_acc_test))
                    print("\nTest loss      : {:1.4f}, Test accuracy      : {:1.4f}\n".
                            format(epoch_loss_test[-1], epoch_acc_test[-1]))
                else:
                    print("Test loss: {:1.4f}".format(epoch_loss_test[-1]))
                y_pred_test = np.concatenate([y_pred_test[i].to("cpu").numpy()
                                for i in range(len(y_pred_test))]).reshape(-1)
                y_test = np.concatenate([y_test[i].to("cpu").numpy()
                                for i in range(len(y_test))]).reshape(-1)
                labels = list(y_test)
                if classification:
                    cltest = classification_report(y_test, y_pred_test, digits=3)
        else:
            epoch_loss_test = epoch_loss_train
            epoch_loss_val = epoch_loss_train
        # early stopping
        print("Countdown: {}\n".format(countdown))
        if epoch_loss_val[-1] <= min(epoch_loss_val):
            countdown = patience #start countdown
            if save_to:
              #Checkpoint: I ovewrite models so as to keep the last to trigger the countdown
              # torch.save(model, os.path.join(save_to, fname, 
              #                                "checkpoint" + ".pt"))
              torch.save(model.state_dict(), 
                         os.path.join(save_to, fname, "state_dict"))
        else:
            countdown -= 1
    print("Finished Training!")

    print("\n\n")
    # results to files and pics
    f = open(os.path.join(save_to, fname, fname + ".txt"), "a+")
    f.write("Supervision: "+ str(supervision) +"\n\n"+
            "Learning rate: "+ str(lr) + "\n\n" +
            "L2 regularizer: "+ str(regularizer) + "\n\n" +
            "Classifier dropout: "+ str(dropout) + "\n\n" +
            "Patch size: "+ str(patch_size) + "\n\n" +
            "Transforms applied:\n\n" + str(transform) + "\n\n" +
            "Batch size:\n\n" + str(inputs.size(0)) + "\n\n" +
            "Premodel:\n\n" + str(pre_model) + "\n\n" +
            "Model:\n\n" + str(model) + "\n\n")
    if supervision:
        f.write("Validation Classification Report(last epoch): \n\n {}\n\n".format(clval))
        f.write("Test Classification Report(last epoch): \n\n {}".format(cltest))
    f.close()
    plt1 = plots(epoch_loss_train, epoch_loss_val, epoch_loss_test,
                 metric="loss", save_to=os.path.join(save_to, fname, fname),
                 epoch=epoch, y_limit=y_limit)
    if classification:
        plt2 = plots(epoch_acc_train, epoch_acc_val, epoch_acc_test,
                     metric="accuracy", save_to=os.path.join(save_to, fname, fname),
                     epoch=epoch, y_limit=y_limit)
    return


# Plot Train / Validation / Test Loss and Acc

def plots(train, val, test, save_to, epoch, y_limit, metric="loss"):
    plt.rcParams["figure.figsize"] = (10,4)
    plt.figure()
    plt.title("Relative "+str(metric))
    plt.plot(list(range(1, epoch + 1)), train, label='Training set')
    plt.plot(list(range(1, epoch + 1)), val,  label='Validation set')
    plt.plot(list(range(1, epoch + 1)), test,  label='Test set')
    if metric == "loss":
      plt.scatter(np.argmin(np.array(val))+1,
                  min(val), color="red", label="Minimum validation loss")
      plt.scatter(np.argmin(np.array(test))+1,
                  min(test), color="green", label="Minimum test loss")
    # max val accuracy point
    else:
      maxaccv=max(val)
      plt.scatter(np.argmax(np.array(val))+1,
                 maxaccv, color="red", label="Maximum validation accuracy="+
                 str(np.around(maxaccv,3)))
    # max test accuracy point
      maxacct = max(test)
      plt.scatter(np.argmax(np.array(test))+1,
                  maxacct, color="green", label="Maximum test accuracy="+
                  str(np.around(maxacct,3)))
    plt.grid()
    if y_limit==0:
      y_limit = 0.6 if metric=="loss" else 1.0
    else:
      y_limit = y_limit if metric=="loss" else 1.0
    plt.ylim(0, y_limit)
    plt.legend(fancybox=True)
    if save_to:
      plt.savefig(save_to + "_" + str(metric) + ".png")
    plt.show()
