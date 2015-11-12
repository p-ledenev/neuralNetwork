package model;

import methods.TrainingBuilder;
import org.encog.Encog;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.specific.CSVNeuralDataSet;
import org.encog.ml.train.BasicTraining;
import org.encog.neural.networks.*;
import org.encog.util.csv.CSVFormat;

/**
 * Created by DiKey on 14.08.2015.
 */
public class Runner {

    //public static String dataPath = "f:\\Teddy\\Alfa\\java\\v1.0\\neuralNetwork\\data\\";
    public static String dataPath = "d:\\Projects\\Alfa\\java\\nn\\neuralNetwork\\data\\";
    //public static String dataPath = "./";

    public static void main(String[] args) throws Throwable {

        String fileName = "usd_app";
        String testFileName = "usd_test";

        Log.info("Training data: " + fileName);

        TrainingBuilder trainingBuilder;
        //trainingBuilder = TrainingBuilderFactory.createRBFTrainer();
        trainingBuilder = TrainingBuilderFactory.createResilientPropagationTrainer();
        //trainingBuilder = TrainingBuilderFactory.createNelderMeadTrainer();
        //trainingBuilder = TrainingBuilderFactory.createLevenbergMarquardtTrainer();
        //trainingBuilder = TrainingBuilderFactory.createPNNTrainer();
        //trainingBuilder = TrainingBuilderFactory.createScaledConjugateGradientTrainer();
        //trainingBuilder = TrainingBuilderFactory.createBayesianTrainer();
        //trainingBuilder = TrainingBuilderFactory.createSVNSearchTrainer();

        MLDataSet trainingSet = new CSVNeuralDataSet(dataPath + fileName + ".txt", 6, 3, false, new CSVFormat('.', ';'), false);
        MLDataSet testSet = new CSVNeuralDataSet(dataPath + testFileName + ".txt", 6, 3, false, new CSVFormat('.', ';'), false);

        trainingBuilder.setTrainingDataTitle(fileName);
        BasicTraining training = trainingBuilder.build(trainingSet);
        training.setError(1.);

        Log.info("Starting training");
        for (int i = 1; training.getError() > 0.001; i++) {
            training.iteration();

            if (i % 1000 == 0) {
                BasicNetwork network = (BasicNetwork) trainingBuilder.getNetwork();
                Log.info("Iteration: " + i + "; training error: " + Round.toAmount(training.getError(), 6) +
                        "; test error: " + Round.toAmount(network.calculateError(testSet), 6));
            }

            if (i % 20000 == 0) {
                Log.info("Saving network");
                training.finishTraining();
                trainingBuilder.saveNetwork();
            }
        }

        Encog.getInstance().shutdown();
    }
}
