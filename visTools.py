import numpy as np
import matplotlib 
matplotlib.use('Agg') # To stop errors on goliath
import matplotlib.pyplot as plt

IMAGE_SIZE = 128

def show_preds(filename):
    """ Open a pickle file containing predictions and display them asking the
        user for input
    """
    dataset = np.load(filename)
    if 'ins' not in dataset.files or 'preds' not in dataset.files:
        raise Exception("Dataset missing ins or preds, may not be valid")
    ins = dataset['ins']
    preds = dataset['preds']
    assert ins.shape == preds.shape
    
    raise Exception("Function NOT Finished")

    cur_img = 0        

    while True:
        usr = input("Action (n/b/q):")
        if usr == 'n':
            pass
        elif usr == 'b':
            pass
        elif usr == 'q':
            break
        else:
            print("Unreccognised command:", usr)

def preds2png(filename, outdir, pred_size=128):
    """ Given a saved file convert in/prediction pairs to pngs and save to
        outdir
        Preconditon: Assumes outdir exists and is writeable
                    filename is valid and readable
    """
     
    dataset = np.load(filename)
    if 'ins' not in dataset.files or 'preds' not in dataset.files:
        raise Exception("Dataset missing ins or preds, may not be valid")
    ins = dataset['ins']
    preds = dataset['preds']
    del dataset
    assert ins.shape == preds.shape
    
    batch_size = ins.shape[0]
    
    for i in range(batch_size):
        in_img = ins[i, :].reshape(pred_size, pred_size)
        pred_img = preds[i, :].reshape(pred_size, pred_size)

        plt.imshow(in_img, cmap='gray')  # Save input img
        plt.savefig(outdir + str(i) + "in.png")
    
        plt.imshow(pred_img, cmap='gray')  # save prediction
        plt.savefig(outdir + str(i) + "out.png")
        
        # When visualising weights can use something like
        # http://stackoverflow.com/questions/11775354/how-can-i-display-a-np-array-with-pylab-imshow
        # im = plt.imshow(arr, cmap='hot')
        # plt.colorbar(im, orientation='horizontal')
        # plt.show()        
        
        # multiple images in one figure
        # http://stackoverflow.com/questions/17111525/how-to-show-multiple-images-in-one-figure        



def write_preds(batch_data, batch_labels, preds, outdir, img_dim):
    """ Given the input, label and prediction write images to outdir consisting
        of single samples of all of the above
    
    """
    num = batch_data.shape[0]

    for cur_img in range(num):
        print("Saving image:", cur_img)
        iimg = batch_data[cur_img, :].reshape(img_dim, img_dim)
        limg = batch_labels[cur_img, :].reshape(img_dim, img_dim)
        pimg = preds[cur_img, :].reshape(img_dim, img_dim)

        num_images = 3
        if num_images == 3:
            fig, ax = plt.subplots(1, num_images, figsize=(6,2),)        
        else:
            fig, ax = plt.subplots(1, num_images, figsize=(4,2),)

        fig.subplots_adjust(hspace=0.3, wspace=0.05)

        titles = ['Input', 'Ground Truth', 'Prediction']
        imgs = [iimg, limg, pimg]
        for i in range(3):
            title = titles[i]
            img = imgs[i]
            iplt = ax.flat[i].imshow(img, cmap='gray', interpolation='none')  
            iplt.set_clim(0.0, 1.0)
            ax.flat[i].axis('off')
            ax.flat[i].set_title(title)

        plt.savefig(outdir + str(cur_img) + ".eps")
        plt.clf()
        plt.close(fig)

