package model;

import org.apache.log4j.Logger;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.specific.CSVNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.csv.CSVFormat;

import java.io.File;

/**
 * Created by DiKey on 14.08.2015.
 */
public class Runner {

    public static String dataPath = "f:\\Teddy\\Alfa\\java\\v1.0\\neuralNetwork\\data\\";
    public static String networkFileName = "network.eg";
    public static Logger logger = Logger.getLogger(Runner.class);

    public static void main(String[] args) throws Throwable {
        BasicNetwork network;

        File file = new File(dataPath + networkFileName);
        if (file.exists()) {
            logger.info("Reading from file");
            network = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File(dataPath + networkFileName));
        }
        else {
            logger.info("Create new network");
            network = createNewNetwork();
        }

        MLDataSet trainingSet = new CSVNeuralDataSet(dataPath + "usd_2010.txt", 50, 1, false, new CSVFormat('.', ';'), false);

        ResilientPropagation training = new ResilientPropagation(network, trainingSet);
        training.setError(1.);

        for (int i = 0; i < 10; i++) {
            training.iteration();
            logger.info("Iteration: " + i + "; error: " + training.getError());
        }

        training.finishTraining();

        logger.info("Saving network");
        EncogDirectoryPersistence.saveObject(new File(dataPath + networkFileName), network);

        Encog.getInstance().shutdown();
    }

    private static BasicNetwork createNewNetwork() {

        BasicNetwork network = new BasicNetwork();

        network.addLayer(new BasicLayer(new ActivationTANH(), true, 50));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 100));
        network.addLayer(new BasicLayer(new ActivationTANH(), false, 1));

        network.getStructure().finalizeStructure();
        network.reset();

        return network;
    }
}
