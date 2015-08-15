package model;

import org.apache.log4j.Logger;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.ResilientPropagation;
import org.neuroph.util.TransferFunctionType;

import java.io.File;

/**
 * Created by DiKey on 14.08.2015.
 */
public class Runner {

    public static String dataPath = "f:\\Teddy\\Alfa\\java\\v1.0\\neuralNetwork\\data\\";

    public static Logger logger = Logger.getLogger(Runner.class);

    public static void main(String[] args) throws Throwable {

        NeuralNetwork net;
        File netFile = new File(dataPath + "network.data");
        if (netFile.exists()) {
            logger.info("Init from file");
            net = NeuralNetwork.createFromFile(netFile);
        } else {
            logger.info("Create new network");
            net = new MultiLayerPerceptron(TransferFunctionType.TANH, 50, 100, 1);
        }

        logger.info("Start reading");
        DataSet trainingSet = DataSet.createFromFile(dataPath + "usd_2010.txt", 50, 1, ";", false);
        logger.info("Reading done");

        ResilientPropagation learningRule = new ResilientPropagation();
        learningRule.setMaxError(0.001);
        learningRule.setMaxIterations(1000);

        learningRule.addListener(new EventPrinter());

        logger.info("Start learning");
        net.learnInNewThread(trainingSet, learningRule);
        synchronized (net) {
            net.wait();
        }
        logger.info("Learning finished");
    }
}
