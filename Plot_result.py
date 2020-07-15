import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    eval_psnrarray = np.load("eval_psnr_array.npy")
    train_psnrarray = np.load("train_psnr_array.npy")
    print(eval_psnrarray)
    print(train_psnrarray)

    x= range(len(eval_psnrarray))
    y_eval = eval_psnrarray
    y_train = train_psnrarray
    plt.plot(x,y_train)
    plt.plot(x,y_eval)
    plt.legend(['train PSNR','evaluation PSNR'])
    plt.title('average PSNR about train and evaluation')
    plt.savefig("PSNR_plot.png",dpi = 500)
    #plt.show()