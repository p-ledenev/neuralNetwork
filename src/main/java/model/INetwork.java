package model;

import lombok.AllArgsConstructor;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.train.BasicTraining;
import org.encog.persist.EncogDirectoryPersistence;

import java.io.File;

/**
 * Created by ledenev.p on 17.08.2015.
 */

@AllArgsConstructor
public abstract class INetwork<TTraining extends BasicTraining, TNetwork extends BasicML> {

    private TTraining training;

    public void iteration() {
        training.iteration();
    }

    public double getError() {
        return training.getError();
    }

    public abstract void setThreadCount(int count);

    public void finishTraining() {
        training.finishTraining();
    }

    public void initFromFile(MLDataSet trainingSet) {

        TNetwork network = (TNetwork) EncogDirectoryPersistence.loadObject(new File(fileName));
        setNetwork(network);
        training.setTraining(trainingSet);
    }

    protected abstract void setNetwork(TNetwork network);
}
