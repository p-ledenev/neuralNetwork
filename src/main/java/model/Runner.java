package model;

import org.apache.log4j.*;
import org.encog.*;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.*;
import org.encog.ml.data.specific.*;
import org.encog.neural.networks.*;
import org.encog.neural.networks.layers.*;
import org.encog.neural.networks.training.propagation.*;
import org.encog.neural.networks.training.propagation.resilient.*;
import org.encog.persist.*;
import org.encog.util.csv.*;

import java.io.*;

/**
 * Created by DiKey on 14.08.2015.
 */
public class Runner {

    //public static String dataPath = "f:\\Teddy\\Alfa\\java\\v1.0\\neuralNetwork\\data\\";
    public static String dataPath = "d:\\Projects\\Alfa\\java\\nn\\neuralNetwork\\data\\";

    public static String networkFileName = "network.eg";
    public static Logger logger = Logger.getLogger(Runner.class);

    public static void main(String[] args) throws Throwable {
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
