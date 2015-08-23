package networks;

import lombok.*;
import model.*;
import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.persist.*;

import java.io.*;

/**
 * Created by ledenev.p on 17.08.2015.
 */

public abstract class NetworkBuilder {

    @Setter
    private String trainingSetName;

    @Getter
    protected BasicML network;

    protected boolean networkExist() {
        File file = new File(getFileName());
        return file.exists();
    }

    public BasicML build(MLDataSet trainingSet) {

        if (networkExist()) {
            read();
        } else {
            Log.info("Creating new " + getName() + " network");
            network = createNetwork(trainingSet);
        }

        return network;
    }

    public void saveNetwork() {
        EncogDirectoryPersistence.saveObject(new File(getFileName()), network);
    }

    private String getFileName() {
        return Runner.dataPath + getName() + "_" + trainingSetName + ".eg";
    }

    protected abstract String getName();

    protected abstract BasicML createNetwork(MLDataSet trainingSet);

    public BasicML read() {
        Log.info("Reading network " + getName() + " from file");
        network = (BasicML) EncogDirectoryPersistence.loadObject(new File(getFileName()));

        return network;
    }
}
