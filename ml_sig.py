__author__ = 'sergio'

def kmeans_clust(filename, melody_pitch, melody_confidence):
# Ground truth (manual midi transcritpion of the melodyes) completed Jul 2015
    import essentia_extractor_sig
    from sklearn import cluster as cl
    import matplotlib.pyplot as plt
    import numpy as np

    kmeans_run = cl.KMeans(n_clusters = 2)
    mfccs = essentia_extractor_sig.MFCCs(filename)

    plot_again = 1

    while plot_again:
        # arrange only mfcc values in a array
        X=[]
        data_in = input('input data as only_mfccs(1), mfccs_and_confidence(2), mfccs_and_pitch(3), mfccs_confidence_and_pitch(4):')
        for i in range (len(mfccs)):
            if data_in == 1:
                X.append(mfccs[i][1])
            if data_in == 2:
                X.append(np.append(mfccs[i][1],[melody_confidence[i]]))
            if data_in == 3:
                X.append(np.append(mfccs[i][1],[melody_pitch[i]]))
            if data_in == 4:
                X.append(np.append(mfccs[i][1],[melody_confidence[i],melody_pitch[i]]))

        lables = kmeans_run.fit_predict(X)

        plt.figure(1)
        plt.subplot(211)
        plt.plot(melody_pitch)
        plt.subplot(212)
        plt.plot(lables)


        matrix=np.matrix(X).transpose()
        fig=plt.figure(2)
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.spectral)
        plt.colorbar()
        plt.show()

        plot_again = input('plot again?:')


    asd=1


