package model;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.specific.CSVNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.csv.CSVFormat;

import java.io.File;

/**
 * Created by ledenev.p on 17.08.2015.
 */
public class Perceptron  implements INetwork{

    BasicNetwork network;

    File file = new File(dataPath + networkFileName);
    if (file.exists()) {
        logger.info("Reading from file");
        network = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File(dataPath + networkFileName));
    } else {
        logger.info("Create new network");
        network = createNewNetwork();
    }

    MLDataSet trainingSet = new CSVNeuralDataSet(dataPath + "usd_2010.txt", 50, 1, false, new CSVFormat('.', ';'), false);

    Propagation training;
    training = new ResilientPropagation(network, trainingSet, 0.2, 100);
    //training = new QuickPropagation(network, trainingSet);
    //training = new ManhattanPropagation(network, trainingSet, 0.2);
    //training = new ScaledConjugateGradient(network, trainingSet);

    training.setThreadCount(2);

    training.setError(1.);

    for (int i = 1; training.getError() > 0.01; i++) {
        training.iteration();
        logger.info("Iteration: " + i + "; error: " + training.getError());

        if (i % 500 == 0) {
            logger.info("Saving network");
            EncogDirectoryPersistence.saveObject(new File(dataPath + networkFileName + "." + i), network);
        }
    }

    training.finishTraining();

    logger.info("Saving network");
    EncogDirectoryPersistence.saveObject(new File(dataPath + networkFileName), network);

    Encog.getInstance().shutdown();
}

    private static BasicNetwork createNewNetwork() {

        BasicNetwork network = new BasicNetwork();

        network.addLayer(new BasicLayer(new ActivationTANH(), true, 50));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 75));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 50));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 25));
        network.addLayer(new BasicLayer(new ActivationTANH(), false, 1));

        network.getStructure().finalizeStructure();
        network.reset();

        return network;
    }
}
